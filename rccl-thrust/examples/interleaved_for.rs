use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{DeviceBufferExt, DeviceRangeExt, ForFrame, InterleavedForRange, pack_for_into, pack_for_into_async, pack_for_workspace, transform_into, lower_bound};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let exec = program.par(&stream)?;
    for (block_rows, lanes) in [(256u32, 8u32), (256, 32), (1024, 32)] {
        for bits in 0..=64u32 {
            let mask = if bits == 0 { 0 } else { u64::MAX >> (64 - bits) };
            for rows in [0, 1, lanes as usize - 1, lanes as usize, lanes as usize + 1, block_rows as usize - 1, block_rows as usize, block_rows as usize + 1, block_rows as usize * 2 + 1] {
                let base = u64::MAX - mask;
                let input: Vec<_> = (0..rows).map(|r| base + if r % block_rows as usize == 0 { 0 } else if r % block_rows as usize == 1 { mask } else { (r as u64).wrapping_mul(0x9e3779b97f4a7c15) & mask }).collect();
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let blocks = rows.div_ceil(block_rows as usize);
                let mut payload = DeviceBuffer::from_host(&stream, &vec![0xdeadbeefu32; rows * 2 + blocks * lanes as usize + 1])?;
                let mut frames = DeviceBuffer::from_host(&stream, &vec![ForFrame::default(); blocks])?;
                let mut output = DeviceBuffer::from_host(&stream, &vec![0u64; rows])?;
                let mut workspace = pack_for_workspace(&exec, blocks)?;
                pack_for_into(&exec, source.view().map(|x: u64| x), payload.view_mut(), frames.view_mut(), block_rows, lanes, &mut workspace)?;
                let used = workspace.words_to_host()? as usize;
                let host_frames = frames.to_host_vec(&stream)?;
                let host_payload = payload.to_host_vec(&stream)?;
                assert_eq!(host_payload[used], 0xdeadbeef);
                for (b, frame) in host_frames.iter().enumerate() {
                    let values = &input[b * block_rows as usize..((b + 1) * block_rows as usize).min(rows)];
                    assert_eq!(frame.base, *values.iter().min().unwrap());
                    assert_eq!(frame.bits, 64 - (values.iter().max().unwrap() - frame.base).leading_zeros());
                    let column = rccl_cub::ForColumn { payload: unsafe { host_payload.as_ptr().add(frame.offset as usize) }, base: frame.base, bits: frame.bits, rows: frame.rows, lanes };
                    for (r, &value) in values.iter().enumerate() { assert_eq!(unsafe { column.read(r as u32) }, value); }
                }
                let range = unsafe { InterleavedForRange::new(payload.view(), frames.view(), rows, block_rows, lanes)? };
                transform_into(&exec, range, output.view_mut(), |x: u64| x)?;
                assert_eq!(output.to_host_vec(&stream)?, input);
                // Dropping the completion still completes before output reuse.
                drop(pack_for_into_async(&exec, source.view(), payload.view_mut(), frames.view_mut(), block_rows, lanes, &mut workspace));
            }
        }
        let sorted: Vec<_> = (0..2049u64).map(|x| x / 3 + 7).collect();
        let source = DeviceBuffer::from_host(&stream, &sorted)?;
        let mut payload = DeviceBuffer::from_host(&stream, &vec![0u32; sorted.len() * 2])?;
        let mut frames = DeviceBuffer::from_host(&stream, &vec![ForFrame::default(); sorted.len().div_ceil(block_rows as usize)])?;
        let mut workspace = pack_for_workspace(&exec, frames.len())?;
        pack_for_into(&exec, source.view(), payload.view_mut(), frames.view_mut(), block_rows, lanes, &mut workspace)?;
        let range = unsafe { InterleavedForRange::new(payload.view(), frames.view(), sorted.len(), block_rows, lanes)? };
        let probes: Vec<_> = (0..700u64).collect();
        let probes_device = DeviceBuffer::from_host(&stream, &probes)?;
        let mut positions = DeviceBuffer::from_host(&stream, &vec![0u64; probes.len()])?;
        lower_bound(&exec, range, probes_device.view(), positions.view_mut())?;
        assert_eq!(positions.to_host_vec(&stream)?, probes.iter().map(|p| sorted.partition_point(|v| v < p) as u64).collect::<Vec<_>>());
    }
    println!("interleaved FOR pack/range/search CPU oracle: PASS");
    Ok(())
}
