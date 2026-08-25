use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use std::error::Error;

mod gpu_support;

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use cuda_device::{kernel, thread};

    #[kernel]
    pub unsafe fn find_non_trivial_runs(
        keys: *const u32,
        count: u64,
        offsets: *mut u64,
        lengths: *mut u64,
        run_count: *mut u64,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        if count == 0 {
            unsafe { run_count.write(0) };
            return;
        }
        let mut runs = 0u64;
        let mut begin = 0u64;
        let mut previous = unsafe { keys.read() };
        let mut index = 1u64;
        while index <= count {
            let boundary = index == count || unsafe { keys.add(index as usize).read() } != previous;
            if boundary {
                let length = index - begin;
                if length > 1 {
                    unsafe {
                        offsets.add(runs as usize).write(begin);
                        lengths.add(runs as usize).write(length);
                    }
                    runs += 1;
                }
                if index != count {
                    begin = index;
                    previous = unsafe { keys.add(index as usize).read() };
                }
            }
            index += 1;
        }
        unsafe { run_count.write(runs) };
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const COUNT: usize = 1_000_003;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let module = kernels::load(&context)?;
    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    let offsets = DeviceBuffer::<u64>::zeroed(&stream, COUNT)?;
    let lengths = DeviceBuffer::<u64>::zeroed(&stream, COUNT)?;
    let run_count = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    support.fill_low_entropy_u32(&stream, &mut keys, 0x4e4f_4e54_5256_5255)?;
    rccl::sort(&exec, &mut keys)?;
    unsafe {
        module.find_non_trivial_runs(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            keys.cu_deviceptr() as *const u32,
            COUNT as u64,
            offsets.cu_deviceptr() as *mut u64,
            lengths.cu_deviceptr() as *mut u64,
            run_count.cu_deviceptr() as *mut u64,
        )?;
    }
    let run_count = run_count.to_host_vec(&stream)?[0];
    println!("found {run_count} non-trivial GPU runs: PASS");
    Ok(())
}
