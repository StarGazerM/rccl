//! Host/device identity probe for pinned cuCollections XXHash_64 vectors.

#[path = "../tests/support/xxhash64_vectors.rs"]
mod vectors;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use rccl_cuco::XxHash64;

const THREADS: u32 = 256;
const STRIDE: usize = 72;

#[cuda_device::cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            offsets.len() >= output.len(),
            lengths.len() >= output.len(),
            seeds.len() >= output.len(),
            data.len() >= output.len() * 72,
        ),
    )]
    pub fn xxhash64_vectors(
        data: &[u8],
        offsets: &[u32],
        lengths: &[u32],
        seeds: &[u64],
        mut output: DisjointSlice<u64>,
    ) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            // SAFETY: the launch contract gives every output index matching
            // metadata. Host construction keeps each range inside its STRIDE.
            let offset = unsafe { *offsets.get_unchecked(position) } as usize;
            let length = unsafe { *lengths.get_unchecked(position) } as usize;
            let seed = unsafe { *seeds.get_unchecked(position) };
            let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr().add(offset), length) };
            *slot = XxHash64::new(seed).hash_bytes(bytes);
        }
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (rows.len() >= 1, values.len() >= 1, output.len() >= 2),
    )]
    pub fn xxhash64_values(rows: &[[u32; 4]], values: &[u64], mut output: DisjointSlice<u64>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = if position == 0 {
                // SAFETY: the launch contract guarantees the first row.
                let row = unsafe { rows.get_unchecked(0) };
                XxHash64::new(0x9e37_79b9_7f4a_7c15).hash_value(row)
            } else {
                // SAFETY: the output length limits this branch to position one
                // and the launch contract guarantees the first primitive.
                let value = unsafe { values.get_unchecked(0) };
                XxHash64::new(0x9e37_79b9_7f4a_7c15).hash_value(value)
            };
        }
    }
}

struct Case {
    kind: &'static str,
    bytes: Vec<u8>,
    seed: u64,
    expected: u64,
    unaligned: bool,
}

fn cases() -> Vec<Case> {
    let pattern = vectors::pattern_bytes();
    let mut cases = Vec::with_capacity(134);
    for length in 0..=63 {
        let seed = vectors::SEEDS[length % vectors::SEEDS.len()];
        let expected = vectors::TAIL_HASHES[length];
        cases.push(Case {
            kind: "tail-aligned",
            bytes: pattern[..length].to_vec(),
            seed,
            expected,
            unaligned: false,
        });
        cases.push(Case {
            kind: "tail-unaligned",
            bytes: pattern[..length].to_vec(),
            seed,
            expected,
            unaligned: true,
        });
    }

    cases.push(Case {
        kind: "u32",
        bytes: 0x1234_5678u32.to_le_bytes().to_vec(),
        seed: 0,
        expected: vectors::U32_HASH,
        unaligned: false,
    });
    cases.push(Case {
        kind: "u64",
        bytes: 0x0123_4567_89ab_cdefu64.to_le_bytes().to_vec(),
        seed: 0x9e37_79b9_7f4a_7c15,
        expected: vectors::U64_HASH,
        unaligned: false,
    });

    let rows: [(&'static str, Vec<u32>, u64, u64); 4] = [
        ("hisa-row-1", vec![0], 0, vectors::HISA_ROW_HASHES[0]),
        ("hisa-row-2", vec![1, 2], 1, vectors::HISA_ROW_HASHES[1]),
        (
            "hisa-row-4",
            vec![0, 1, 0x1234_5678, 0xffff_ffff],
            0x9e37_79b9_7f4a_7c15,
            vectors::HISA_ROW_HASHES[2],
        ),
        (
            "hisa-row-7",
            vec![7, 0, 42, 0x8000_0000, 3, 99, 0xffff_ffff],
            0,
            vectors::HISA_ROW_HASHES[3],
        ),
    ];
    for (kind, row, seed, expected) in rows {
        let mut bytes = Vec::with_capacity(row.len() * 4);
        for column in row {
            bytes.extend_from_slice(&column.to_le_bytes());
        }
        cases.push(Case {
            kind,
            bytes,
            seed,
            expected,
            unaligned: false,
        });
    }
    cases
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = cases();
    let mut data = vec![0u8; cases.len() * STRIDE];
    let mut offsets = Vec::with_capacity(cases.len());
    let mut lengths = Vec::with_capacity(cases.len());
    let mut seeds = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let offset = index * STRIDE + usize::from(case.unaligned);
        data[offset..offset + case.bytes.len()].copy_from_slice(&case.bytes);
        offsets.push(offset as u32);
        lengths.push(case.bytes.len() as u32);
        seeds.push(case.seed);
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = unsafe { kernels::load(&context)? };
    let data = DeviceBuffer::from_host(&stream, &data)?;
    let offsets = DeviceBuffer::from_host(&stream, &offsets)?;
    let lengths = DeviceBuffer::from_host(&stream, &lengths)?;
    let seeds = DeviceBuffer::from_host(&stream, &seeds)?;
    let mut output = DeviceBuffer::<u64>::zeroed(&stream, cases.len())?;
    let launch = LaunchConfig1D::new(1, THREADS, 0);
    let prepared = program.prepare_xxhash64_vectors(launch)?;
    program.xxhash64_vectors(
        &stream,
        &prepared,
        &data,
        &offsets,
        &lengths,
        &seeds,
        &mut output,
    )?;

    let workload_rows = DeviceBuffer::from_host(&stream, &[[0, 1, 0x1234_5678, 0xffff_ffff]])?;
    let workload_values = DeviceBuffer::from_host(&stream, &[0x0123_4567_89ab_cdefu64])?;
    let mut workload_output = DeviceBuffer::<u64>::zeroed(&stream, 2)?;
    let workload_launch = LaunchConfig1D::new(1, 32, 0);
    let workload_prepared = program.prepare_xxhash64_values(workload_launch)?;
    program.xxhash64_values(
        &stream,
        &workload_prepared,
        &workload_rows,
        &workload_values,
        &mut workload_output,
    )?;
    assert_eq!(
        workload_output.to_host_vec(&stream)?,
        [vectors::HISA_ROW_HASHES[2], vectors::U64_HASH]
    );
    let output = output.to_host_vec(&stream)?;

    for (case, actual) in cases.iter().zip(output) {
        assert_eq!(
            actual, case.expected,
            "{} seed {:#018x}",
            case.kind, case.seed
        );
        println!(
            "{} {} {:#018x} {:#018x}",
            case.kind,
            case.bytes.len(),
            case.seed,
            actual
        );
    }
    Ok(())
}
