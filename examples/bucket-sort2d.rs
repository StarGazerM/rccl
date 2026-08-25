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
    pub unsafe fn bucket_bounds(keys: *const u32, count: u64, bucket: u32, bounds: *mut u64) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut first = 0u64;
        let mut len = count;
        while len != 0 {
            let step = len / 2;
            let middle = first + step;
            if unsafe { keys.add(middle as usize).read() } < bucket {
                first = middle + 1;
                len -= step + 1;
            } else {
                len = step;
            }
        }
        let lower = first;
        len = count - lower;
        while len != 0 {
            let step = len / 2;
            let middle = first + step;
            if unsafe { keys.add(middle as usize).read() } <= bucket {
                first = middle + 1;
                len -= step + 1;
            } else {
                len = step;
            }
        }
        unsafe {
            bounds.write(lower);
            bounds.add(1).write(first);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const COUNT: usize = 1_000_000;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let module = kernels::load(&context)?;
    let mut bucket_indices = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    let mut point_ids = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    let bounds = DeviceBuffer::<u64>::zeroed(&stream, 2)?;

    // The low-entropy generator assigns every device-resident point id to one
    // of sixteen buckets; no point or bucket array is constructed on the CPU.
    support.fill_low_entropy_u32(&stream, &mut bucket_indices, 0x4255_434b_4554_3244)?;
    support.fill_iota_u32(&stream, &mut point_ids)?;
    rccl::stable_sort_by_key(&exec, &mut bucket_indices, &mut point_ids)?;
    assert_eq!(
        support.validate_stable_u32_pairs(&stream, &bucket_indices, &point_ids, rccl::Less,)?,
        0
    );
    unsafe {
        module.bucket_bounds(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            bucket_indices.cu_deviceptr() as *const u32,
            COUNT as u64,
            7,
            bounds.cu_deviceptr() as *mut u64,
        )?;
    }
    let bounds = bounds.to_host_vec(&stream)?;
    println!(
        "bucket 7 contains {} GPU-resident point ids: PASS",
        bounds[1] - bounds[0]
    );
    Ok(())
}
