use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, equal_range_by_async, lower_bound_by_async, upper_bound_by_async,
};
use std::error::Error;
use std::time::Instant;

const WARMUPS: usize = 10;
const SAMPLES: usize = 31;
const SEED: usize = 19;

fn main() -> Result<(), Box<dyn Error>> {
    let cold = Instant::now();
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    stream.synchronize()?;
    println!("setup implementation=rccl context_program_ms={:.6}", cold.elapsed().as_secs_f64() * 1000.0);
    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);

    for (size, haystack_len, needle_len) in [("small", 1024, 256), ("large", 1 << 20, 1 << 18)] {
        for dataset in ["uniform", "duplicates", "misses"] {
            for descending in [false, true] {
                let host_start = Instant::now();
                let repeats = if dataset == "duplicates" { 32 } else { 1 };
                let distinct = haystack_len / repeats;
                let mut host_haystack: Vec<u64> = (0..haystack_len)
                    .map(|index| (2 * (index / repeats) + 2) as u64)
                    .collect();
                if descending {
                    host_haystack.reverse();
                }
                let host_needles: Vec<u64> = (0..needle_len)
                    .map(|index| {
                        let key = (2 * ((index * 65_537 + SEED) % distinct) + 2) as u64;
                        key + u64::from(dataset == "misses" && index % 8 != 0)
                    })
                    .collect();
                let host_setup_ms = host_start.elapsed().as_secs_f64() * 1000.0;
                let compare = move |left: &u64, right: &u64| {
                    if descending { right.cmp(left) } else { left.cmp(right) }
                };
                let expected_begins: Vec<u64> = host_needles.iter().map(|needle| {
                    host_haystack.partition_point(|key| compare(key, needle).is_lt()) as u64
                }).collect();
                let expected_ends: Vec<u64> = host_needles.iter().map(|needle| {
                    host_haystack.partition_point(|key| !compare(needle, key).is_lt()) as u64
                }).collect();

                let setup = Instant::now();
                let haystack = DeviceBuffer::from_host(&stream, &host_haystack)?;
                let needles = DeviceBuffer::from_host(&stream, &host_needles)?;
                let mut begins = DeviceBuffer::<u64>::zeroed(&stream, needle_len)?;
                let mut ends = DeviceBuffer::<u64>::zeroed(&stream, needle_len)?;
                stream.synchronize()?;
                let device_setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
                let order = if descending { "descending" } else { "ascending" };
                println!("setup implementation=rccl size={size} dataset={dataset} order={order} host_ms={host_setup_ms:.6} allocation_upload_ms={device_setup_ms:.6} storage_bytes={}", (haystack_len + 3 * needle_len) * 8);

                for algorithm in ["lower_bound", "upper_bound", "equal_range"] {
                    let mut values = Vec::with_capacity(SAMPLES);
                    // Keep the production async submission and completion path in every sample.
                    // Device-buffer allocation, upload and CPU validation are outside the events.
                    // Both benchmarks create/record events around each submission identically.
                    macro_rules! measure {
                        ($submit:expr) => {{
                            for _ in 0..WARMUPS {
                                $submit.wait()?;
                            }
                            for _ in 0..SAMPLES {
                                let start = stream.record_event(timing)?;
                                let pending = $submit;
                                let end = stream.record_event(timing)?;
                                end.synchronize()?;
                                values.push(start.elapsed_ms(&end)?);
                                pending.wait()?;
                            }
                        }};
                    }
                    match algorithm {
                        "lower_bound" => {
                            measure!(lower_bound_by_async(&execution, haystack.view(), needles.view(), begins.view_mut(), compare));
                            assert_eq!(begins.to_host_vec(&stream)?, expected_begins);
                        }
                        "upper_bound" => {
                            measure!(upper_bound_by_async(&execution, haystack.view(), needles.view(), begins.view_mut(), compare));
                            assert_eq!(begins.to_host_vec(&stream)?, expected_ends);
                        }
                        _ => {
                            measure!(equal_range_by_async(&execution, haystack.view(), needles.view(), begins.view_mut(), ends.view_mut(), compare));
                            assert_eq!(begins.to_host_vec(&stream)?, expected_begins);
                            assert_eq!(ends.to_host_vec(&stream)?, expected_ends);
                        }
                    }
                    values.sort_by(f32::total_cmp);
                    let raw = values.iter().map(|value| format!("{value:.9}")).collect::<Vec<_>>().join(",");
                    println!("implementation=rccl algorithm={algorithm} size={size} dataset={dataset} order={order} key=u64 index=u64 haystack={haystack_len} needles={needle_len} seed={SEED} warmups={WARMUPS} samples={SAMPLES} stream=default resets=none best_ms={:.9} median_ms={:.9} sample_ms={raw}", values[0], values[SAMPLES / 2]);
                }
            }
        }
    }
    Ok(())
}
