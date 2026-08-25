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
    pub unsafe fn find_mode(keys: *const u32, count: u64, result: *mut u64) {
        if thread::index_1d().get() != 0 {
            return;
        }
        if count == 0 {
            unsafe {
                result.write(0);
                result.add(1).write(0);
            }
            return;
        }
        let mut best_key = unsafe { keys.read() };
        let mut best_count = 1u64;
        let mut current_key = best_key;
        let mut current_count = 1u64;
        let mut index = 1u64;
        while index < count {
            let key = unsafe { keys.add(index as usize).read() };
            if key == current_key {
                current_count += 1;
            } else {
                if current_count > best_count {
                    best_key = current_key;
                    best_count = current_count;
                }
                current_key = key;
                current_count = 1;
            }
            index += 1;
        }
        if current_count > best_count {
            best_key = current_key;
            best_count = current_count;
        }
        unsafe {
            result.write(best_key as u64);
            result.add(1).write(best_count);
        }
    }

    #[kernel]
    pub unsafe fn validate_mode(
        keys: *const u32,
        count: u64,
        result: *const u64,
        errors: *mut u32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mode_key = unsafe { result.read() } as u32;
        let mode_count = unsafe { result.add(1).read() };
        let mut occurrences = 0u64;
        let mut maximum_run = 0u64;
        let mut current_run = 0u64;
        let mut previous = 0u32;
        let mut index = 0u64;
        while index < count {
            let key = unsafe { keys.add(index as usize).read() };
            if key == mode_key {
                occurrences += 1;
            }
            if index == 0 || key != previous {
                current_run = 1;
                previous = key;
            } else {
                current_run += 1;
            }
            if current_run > maximum_run {
                maximum_run = current_run;
            }
            index += 1;
        }
        let invalid = if mode_count != occurrences || mode_count != maximum_run {
            1
        } else {
            0
        };
        unsafe { errors.write(invalid) };
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
    let result = DeviceBuffer::<u64>::zeroed(&stream, 2)?;
    let errors = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    support.fill_low_entropy_u32(&stream, &mut keys, 0x4d4f_4445)?;
    rccl::sort(&exec, &mut keys)?;
    assert_eq!(support.validate_order(&stream, &keys, rccl::Less)?, 0);
    unsafe {
        module.find_mode(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            keys.cu_deviceptr() as *const u32,
            COUNT as u64,
            result.cu_deviceptr() as *mut u64,
        )?;
        module.validate_mode(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            keys.cu_deviceptr() as *const u32,
            COUNT as u64,
            result.cu_deviceptr() as *const u64,
            errors.cu_deviceptr() as *mut u32,
        )?;
    }
    let mut error = 0u32;
    errors.copy_to_host(&stream, std::slice::from_mut(&mut error))?;
    assert_eq!(error, 0);
    println!("device-resident mode computation: PASS");
    Ok(())
}
