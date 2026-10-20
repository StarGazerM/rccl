//! Stable lexicographic sort of `u32` key columns into a row permutation.
//!
//! Two strategies with the same result, the stable permutation `p` such that
//! `(c0[p[i]], c1[p[i]], …)` is non-decreasing and equal keys keep row order:
//!
//! * [`LexicographicStrategy::RadixColumns`]: one stable LSD radix pair pass
//!   per column, last column first, over only that column's bits; each later
//!   pass gathers its column through the permutation so far.
//! * [`LexicographicStrategy::MergeTuples`]: pack `[u32; K]` rows and run one
//!   stable merge sort of (row, row ID) with the scalarized lexicographic
//!   comparator of the record sorts.
//!
//! [`LexicographicStrategy::choose`] predicts both costs from the item count,
//! column count and per-column bit widths; see its documentation for the
//! calibration. All work is enqueued on the execution's stream without host
//! synchronization; scratch is released stream-ordered.

use crate::{DeviceAlgorithmError, DeviceBufferExt, cuda::Execution, range::Select};
use cuda_core::DeviceBuffer;
use rccl_memory::DeviceAllocator;

/// Largest supported column count (the record-sort comparator's arity).
pub const MAX_LEXICOGRAPHIC_COLUMNS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LexicographicStrategy {
    RadixColumns,
    MergeTuples,
}

impl LexicographicStrategy {
    /// The strategy with the lower predicted time. Calibrated on sm_89 (RTX
    /// 6000 Ada) by `examples/lexicographic_sort_strategies`: uniform keys,
    /// 10^4..6·10^7 items, 1..6 columns of 3..32 bits, pooled stream-ordered
    /// scratch, relative-error least squares. Over those 150 cases it picks
    /// the faster strategy in 144; its choice costs 0.4% more than the best on
    /// average and at most 1.22x (either fixed strategy: up to 5.3x / 5.6x).
    /// With `n` items in millions, `k` columns and `P = Σ ⌈bits/8⌉` radix
    /// passes (ms): `radix = a0·P + a1·P·n + a2·k·n + a3`,
    /// `merge = b0 + b1·n·log2(items) + b2·k·n·log2(items)`.
    pub fn choose(items: usize, bits: &[u32]) -> Self {
        let n = items as f64 / 1e6;
        let k = bits.len() as f64;
        let passes: f64 = bits.iter().map(|bits| bits.div_ceil(8) as f64).sum();
        let log = (items.max(2) as f64).log2();
        let radix = RADIX[0] * passes + RADIX[1] * passes * n + RADIX[2] * k * n + RADIX[3];
        let merge = MERGE[0] + MERGE[1] * n * log + MERGE[2] * k * n * log;
        if merge < radix { Self::MergeTuples } else { Self::RadixColumns }
    }
}

const RADIX: [f64; 4] = [0.024672, 0.022377, 0.023628, 0.018764];
const MERGE: [f64; 3] = [0.114428, 0.000854, 0.008441];

/// Write into `permutation` the stable lexicographic order of the rows of
/// `columns` (most significant first; each value below `2^bits[j]`), using
/// `strategy` or, when `None`, [`LexicographicStrategy::choose`]. Returns the
/// strategy used. Values at or above `2^bits[j]` make the order unspecified
/// (the radix strategy reads only the declared bits).
pub fn sort_columns_lexicographic<A: DeviceAllocator>(
    execution: &Execution<'_, '_, A>,
    columns: &[&DeviceBuffer<u32>],
    bits: &[u32],
    permutation: &mut DeviceBuffer<u32>,
    strategy: Option<LexicographicStrategy>,
) -> Result<LexicographicStrategy, DeviceAlgorithmError> {
    let items = permutation.len();
    if columns.is_empty() || columns.len() > MAX_LEXICOGRAPHIC_COLUMNS {
        return Err(DeviceAlgorithmError::UnsupportedTarget("lexicographic sort supports 1..=6 columns"));
    }
    if bits.len() != columns.len() {
        return Err(DeviceAlgorithmError::LengthMismatch { role: "lexicographic bit widths", expected: columns.len(), actual: bits.len() });
    }
    if let Some(&wide) = bits.iter().find(|&&bits| bits > 32) {
        return Err(DeviceAlgorithmError::InvalidBitRange { begin: 0, end: wide, key_bits: 32 });
    }
    if let Some(column) = columns.iter().find(|column| column.len() != items) {
        return Err(DeviceAlgorithmError::LengthMismatch { role: "lexicographic key column", expected: items, actual: column.len() });
    }
    if items > u32::MAX as usize {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    let strategy = strategy.unwrap_or_else(|| LexicographicStrategy::choose(items, bits));
    if items == 0 {
        return Ok(strategy);
    }
    crate::backend::range_generate(
        execution,
        &rccl_cub::arithmetic_range(0_usize, 1_usize, items),
        &permutation.view_mut(),
        &|row: usize| row as u32,
    )?;
    match strategy {
        LexicographicStrategy::RadixColumns => radix_columns(execution, columns, bits, permutation)?,
        LexicographicStrategy::MergeTuples => match columns.len() {
            1 => merge_tuples::<1, A>(execution, columns, permutation)?,
            2 => merge_tuples::<2, A>(execution, columns, permutation)?,
            3 => merge_tuples::<3, A>(execution, columns, permutation)?,
            4 => merge_tuples::<4, A>(execution, columns, permutation)?,
            5 => merge_tuples::<5, A>(execution, columns, permutation)?,
            _ => merge_tuples::<6, A>(execution, columns, permutation)?,
        },
    }
    Ok(strategy)
}

fn scratch<T: cuda_core::DeviceCopy>(stream: &cuda_core::CudaStream, len: usize) -> Result<DeviceBuffer<T>, DeviceAlgorithmError> {
    let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let context = stream.context().clone();
    if bytes == 0 {
        // SAFETY: a null zero-byte buffer is never dereferenced.
        return Ok(unsafe { DeviceBuffer::from_raw_parts(0, len, context) });
    }
    // SAFETY: a stream-ordered allocation of `bytes` in the stream's context,
    // owned by the buffer and released by `release` on the same stream. Every
    // element is written by a stream-ordered kernel before any read.
    unsafe {
        context.bind_to_thread().map_err(DeviceAlgorithmError::Driver)?;
        let pointer = cuda_core::memory::malloc_async(stream.cu_stream(), bytes).map_err(DeviceAlgorithmError::Driver)?;
        Ok(DeviceBuffer::from_raw_parts(pointer, len, context))
    }
}

fn release<T: cuda_core::DeviceCopy>(stream: &cuda_core::CudaStream, buffer: DeviceBuffer<T>) -> Result<(), DeviceAlgorithmError> {
    // SAFETY: the release is ordered after every enqueued use on this stream.
    unsafe { buffer.drop_async(stream) }.map_err(DeviceAlgorithmError::Driver)
}

fn radix_columns<A: DeviceAllocator>(
    execution: &Execution<'_, '_, A>,
    columns: &[&DeviceBuffer<u32>],
    bits: &[u32],
    permutation: &mut DeviceBuffer<u32>,
) -> Result<(), DeviceAlgorithmError> {
    let stream = execution.stream();
    let items = permutation.len();
    let mut keys = scratch::<u32>(stream, items)?;
    let mut alternate_keys = scratch::<u32>(stream, items)?;
    let mut alternate_order = scratch::<u32>(stream, items)?;
    // Stream-ordered controls lent to the workspace; `launch_radix_init`
    // initializes bins and counters, lookback is zeroed before every pass.
    let counts = rccl_cub::thrust_integration::device_radix_sort_pairs_workspace_counts::<u32, u32>(items);
    let mut bins = scratch::<u64>(stream, counts.bins)?;
    let mut lookback = scratch::<i32>(stream, counts.lookback)?;
    let mut counters = scratch::<i32>(stream, counts.counters)?;
    let mut workspace = rccl_cub::DeviceRadixSortBufferWorkspace::__from_parts(&mut bins, &mut lookback, &mut counters);
    // Which buffer holds the permutation so far; each pass pairs the double
    // buffer around it, so no copy is needed between passes.
    let mut in_permutation = true;
    for column in (0..columns.len()).rev() {
        {
            let source = if in_permutation { permutation.view() } else { alternate_order.view() };
            crate::backend::range_transform(execution, &Select::new(columns[column].view(), source), &keys.view_mut(), &|value| value)?;
        }
        let first_is_current = if in_permutation {
            let mut pairs = rccl_cub::DoubleBuffer::paired((&mut keys, &mut *permutation), (&mut alternate_keys, &mut alternate_order))
                .ok_or(DeviceAlgorithmError::LengthMismatch { role: "lexicographic radix double buffer", expected: items, actual: items })?;
            rccl_cub::device_radix_sort_pairs_buffer(execution.launch(), &mut pairs, 0..bits[column], &mut workspace)?;
            pairs.current_is_first()
        } else {
            let mut pairs = rccl_cub::DoubleBuffer::paired((&mut keys, &mut alternate_order), (&mut alternate_keys, &mut *permutation))
                .ok_or(DeviceAlgorithmError::LengthMismatch { role: "lexicographic radix double buffer", expected: items, actual: items })?;
            rccl_cub::device_radix_sort_pairs_buffer(execution.launch(), &mut pairs, 0..bits[column], &mut workspace)?;
            pairs.current_is_first()
        };
        in_permutation = in_permutation == first_is_current;
    }
    if !in_permutation {
        crate::backend::range_transform(execution, &alternate_order.view(), &permutation.view_mut(), &|value| value)?;
    }
    drop(workspace);
    release(stream, keys)?;
    release(stream, alternate_keys)?;
    release(stream, alternate_order)?;
    release(stream, bins)?;
    release(stream, lookback)?;
    release(stream, counters)
}

fn merge_tuples<const K: usize, A: DeviceAllocator>(
    execution: &Execution<'_, '_, A>,
    columns: &[&DeviceBuffer<u32>],
    permutation: &mut DeviceBuffer<u32>,
) -> Result<(), DeviceAlgorithmError> {
    let stream = execution.stream();
    let items = permutation.len();
    let pointers: [u64; K] = core::array::from_fn(|column| columns[column].cu_deviceptr());
    let mut rows = scratch::<[u32; K]>(stream, items)?;
    crate::backend::range_generate(
        execution,
        &rccl_cub::arithmetic_range(0_usize, 1_usize, items),
        &rows.view_mut(),
        // SAFETY: every pointer is a live column of `items` elements, checked
        // above; reads are ordered before the merge sort on this stream.
        &move |row: usize| core::array::from_fn(|column| unsafe { *(pointers[column] as *const u32).add(row) }),
    )?;
    let partitions = rccl_cub::thrust_integration::device_merge_sort_partition_count::<[u32; K]>(items)?;
    let mut scratch_rows = scratch::<[u32; K]>(stream, items)?;
    let mut scratch_ids = scratch::<u32>(stream, items)?;
    let mut partition_storage = scratch::<usize>(stream, partitions)?;
    let mut workspace =
        rccl_cub::DeviceMergeSortPairsWorkspace::__from_parts(&mut scratch_rows, &mut scratch_ids, &mut partition_storage);
    rccl_cub::device_merge_sort_pairs(execution.launch(), &mut rows, permutation, &mut workspace, crate::record_sort::ascending::<K>)?;
    drop(workspace);
    release(stream, rows)?;
    release(stream, scratch_rows)?;
    release(stream, scratch_ids)?;
    release(stream, partition_storage)
}
