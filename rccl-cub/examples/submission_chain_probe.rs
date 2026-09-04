use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use std::error::Error;

const WARMUPS: usize = 100;
const SAMPLES: usize = 101;
const CHAINS: usize = 100;

#[cuda_device::cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1), requires = (output.len() >= 1))]
    pub fn tiny<const TAG: usize>(output: DisjointSlice<u32>) {
        let _ = TAG;
        if thread::threadIdx_x() == 0 {
            let mut output = rccl_cub::__private::DeviceRegion::new(output);
            let value = output.read(0);
            output.write(0, value.wrapping_add(1));
        }
    }
}

fn median_chain_us(
    stream: &cuda_core::CudaStream,
    mut chain: impl FnMut() -> Result<(), cuda_core::LaunchContractError>,
) -> Result<f32, Box<dyn Error>> {
    for _ in 0..WARMUPS {
        chain()?;
    }
    stream.synchronize()?;
    let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(flags)?;
        for _ in 0..CHAINS {
            chain()?;
        }
        let end = stream.record_event(flags)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? * 1_000.0 / CHAINS as f32);
    }
    samples.sort_by(f32::total_cmp);
    Ok(samples[SAMPLES / 2])
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = unsafe { kernels::load(&context)? };
    let mut output = DeviceBuffer::from_host(&stream, &[0_u32])?;
    let config = LaunchConfig1D::new(1, 1, 0);

    let wrapper_us = median_chain_us(&stream, || {
        let p0 = module.prepare_tiny::<0>(config)?;
        module.tiny::<0>(&stream, &p0, &mut output)?;
        let p1 = module.prepare_tiny::<1>(config)?;
        module.tiny::<1>(&stream, &p1, &mut output)?;
        let p2 = module.prepare_tiny::<2>(config)?;
        module.tiny::<2>(&stream, &p2, &mut output)?;
        let p3 = module.prepare_tiny::<3>(config)?;
        module.tiny::<3>(&stream, &p3, &mut output)
    })?;

    let p0 = module.prepare_tiny::<0>(config)?;
    let p1 = module.prepare_tiny::<1>(config)?;
    let p2 = module.prepare_tiny::<2>(config)?;
    let p3 = module.prepare_tiny::<3>(config)?;
    let prepared_us = median_chain_us(&stream, || {
        module.tiny::<0>(&stream, &p0, &mut output)?;
        module.tiny::<1>(&stream, &p1, &mut output)?;
        module.tiny::<2>(&stream, &p2, &mut output)?;
        module.tiny::<3>(&stream, &p3, &mut output)
    })?;

    stream.synchronize()?;
    let expected = ((WARMUPS + SAMPLES * CHAINS) * 8) as u32;
    assert_eq!(output.to_host_vec(&stream)?, [expected]);
    println!(
        "implementation=rccl-driver kernels=4 chains={} wrapper_median_us={wrapper_us:.6} pre_resolved_median_us={prepared_us:.6}",
        CHAINS
    );
    Ok(())
}
