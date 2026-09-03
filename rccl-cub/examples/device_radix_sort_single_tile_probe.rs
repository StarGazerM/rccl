//! Source-mapped probe for CUB's Policy800 `DeviceRadixSortSingleTileKernel`.
//!
//! The workspace macro materializes zero one-sweep state for these nonempty
//! extents. Passing therefore proves the dedicated single-tile route ran.

use cuda_core::{CudaContext, DeviceBuffer};
use std::{cmp::Ordering, error::Error};

fn field(value: u32, bits: core::ops::Range<u32>) -> u32 {
    let width = bits.end - bits.start;
    (value >> bits.start) & ((1u32 << width) - 1)
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;

    // Policy800 u32 single-tile extent: RegBoundScaling<256, 19, u32>.
    let input = (0..4_864u32)
        .map(|index| index.wrapping_mul(2_654_435_761).rotate_left(index & 31))
        .collect::<Vec<_>>();
    let mut expected = input.clone();
    expected.sort_unstable();
    let mut keys = DeviceBuffer::from_host(&stream, &input)?;
    let mut alternate = DeviceBuffer::<u32>::zeroed(&stream, input.len())?;
    let mut workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, input.len())?;
    let mut buffers = rccl_cub::DoubleBuffer::new(&mut keys, &mut alternate)
        .expect("equal key buffers form one Rust selector");
    rccl_cub::device_radix_sort_keys_buffer(&launch, &mut buffers, 0u32..32u32, &mut workspace)?;
    assert_eq!(buffers.current().to_host_vec(&stream)?, expected);
    assert!(!buffers.current_is_first());

    // A partial window and duplicate fields validate CUB's stable descending
    // rank reversal, rather than a comparison-sort substitute.
    let bits = 3u32..19u32;
    let window_input = (0..4_097u32)
        .map(|index| {
            let duplicate_field = (index % 37) << bits.start;
            duplicate_field | (index.wrapping_mul(97) & 0xfff8_0007)
        })
        .collect::<Vec<_>>();
    let mut window_expected = window_input.clone();
    window_expected.sort_by(|left, right| {
        field(*right, bits.clone())
            .cmp(&field(*left, bits.clone()))
            .then(Ordering::Equal)
    });
    let mut window_keys = DeviceBuffer::from_host(&stream, &window_input)?;
    let mut window_alternate = DeviceBuffer::<u32>::zeroed(&stream, window_input.len())?;
    let mut window_buffers =
        rccl_cub::DoubleBuffer::new(&mut window_keys, &mut window_alternate).unwrap();
    let mut window_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, window_input.len())?;
    rccl_cub::device_radix_sort_keys_buffer_descending(
        &launch,
        &mut window_buffers,
        bits,
        &mut window_workspace,
    )?;
    assert_eq!(
        window_buffers.current().to_host_vec(&stream)?,
        window_expected
    );
    assert!(!window_buffers.current_is_first());

    println!("CUB Policy800 single-tile keys source probe: PASS");
    Ok(())
}
