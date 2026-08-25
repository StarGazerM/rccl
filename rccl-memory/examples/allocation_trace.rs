use rccl_memory::{
    CudaAsyncMemoryResource, CudaMemoryResource, CudaPoolMemoryResource, DeviceAllocator,
    DeviceBlock, PoolConfig, PoolMemoryResource,
};
use std::{
    alloc::Layout,
    error::Error,
    time::{Duration, Instant},
};

const ALIGNMENT: usize = 256;

#[derive(Clone, Copy)]
enum Operation {
    Allocate { slot: usize, size: usize },
    Free { slot: usize },
}

struct TraceResult {
    duration: Duration,
    latencies: Vec<u64>,
    allocation_latencies: Vec<u64>,
    free_latencies: Vec<u64>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }

    fn below(&mut self, maximum: usize) -> usize {
        (self.next() as usize) % maximum
    }
}

fn trace(
    allocation_count: usize,
    maximum_allocation: usize,
    maximum_live_bytes: usize,
    seed: u64,
) -> Vec<Operation> {
    let mut random = Lcg(seed);
    let mut operations = Vec::with_capacity(allocation_count * 2);
    let mut active = Vec::new();
    let mut live_bytes = 0usize;
    let mut allocated = 0usize;
    while allocated < allocation_count {
        let size = random.below(maximum_allocation) + 1;
        let should_allocate = active.is_empty()
            || (random.below(100) < 73 && live_bytes.saturating_add(size) <= maximum_live_bytes);
        if should_allocate {
            let slot = allocated;
            operations.push(Operation::Allocate { slot, size });
            active.push((slot, size));
            live_bytes += size;
            allocated += 1;
        } else {
            let index = random.below(active.len());
            let (slot, size) = active.swap_remove(index);
            operations.push(Operation::Free { slot });
            live_bytes -= size;
        }
    }
    while !active.is_empty() {
        let index = random.below(active.len());
        let (slot, _) = active.swap_remove(index);
        operations.push(Operation::Free { slot });
    }
    operations
}

fn run_trace(
    resource: &(impl DeviceAllocator + ?Sized),
    stream: &cuda_core::CudaStream,
    operations: &[Operation],
    allocation_count: usize,
    capture_latency: bool,
) -> Result<TraceResult, Box<dyn Error>> {
    let mut live: Vec<Option<DeviceBlock>> = vec![None; allocation_count];
    let mut latencies = capture_latency.then(|| Vec::with_capacity(operations.len()));
    let mut allocation_latencies = capture_latency.then(Vec::new);
    let mut free_latencies = capture_latency.then(Vec::new);
    let started = Instant::now();
    for operation in operations {
        let operation_started = capture_latency.then(Instant::now);
        match *operation {
            Operation::Allocate { slot, size } => {
                live[slot] =
                    Some(resource.allocate(Layout::from_size_align(size, ALIGNMENT)?, stream)?);
            }
            Operation::Free { slot } => {
                let block = live[slot].take().expect("trace frees a live allocation");
                unsafe { resource.deallocate(block, stream)? };
            }
        }
        if let (Some(samples), Some(operation_started)) = (&mut latencies, operation_started) {
            let elapsed = operation_started.elapsed().as_nanos() as u64;
            samples.push(elapsed);
            match operation {
                Operation::Allocate { .. } => allocation_latencies
                    .as_mut()
                    .expect("latency capture enabled")
                    .push(elapsed),
                Operation::Free { .. } => free_latencies
                    .as_mut()
                    .expect("latency capture enabled")
                    .push(elapsed),
            }
        }
    }
    stream.synchronize()?;
    Ok(TraceResult {
        duration: started.elapsed(),
        latencies: latencies.unwrap_or_default(),
        allocation_latencies: allocation_latencies.unwrap_or_default(),
        free_latencies: free_latencies.unwrap_or_default(),
    })
}

fn argument(name: &str, default: usize) -> usize {
    std::env::args()
        .skip(1)
        .find_map(|argument| {
            argument
                .strip_prefix(&format!("--{name}="))
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(default)
}

fn resource_name() -> String {
    std::env::args()
        .skip(1)
        .find_map(|argument| argument.strip_prefix("--resource=").map(str::to_owned))
        .unwrap_or_else(|| "cuda_async".to_owned())
}

fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark entry keeps workload controls explicit at the CLI boundary"
)]
fn benchmark(
    resource: &(impl DeviceAllocator + ?Sized),
    resource_name: &str,
    stream: &cuda_core::CudaStream,
    operations: &[Operation],
    allocation_count: usize,
    maximum_allocation: usize,
    maximum_live_bytes: usize,
    warmups: usize,
    repetitions: usize,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..warmups {
        run_trace(resource, stream, operations, allocation_count, false)?;
    }

    let mut elapsed = Vec::with_capacity(repetitions);
    let mut latency = Vec::new();
    let mut allocation_latency = Vec::new();
    let mut free_latency = Vec::new();
    for repetition in 0..repetitions {
        let capture_latency = repetition == repetitions - 1;
        let result = run_trace(
            resource,
            stream,
            operations,
            allocation_count,
            capture_latency,
        )?;
        elapsed.push(result.duration.as_nanos() as u64);
        latency.extend(result.latencies);
        allocation_latency.extend(result.allocation_latencies);
        free_latency.extend(result.free_latencies);
    }
    elapsed.sort_unstable();
    latency.sort_unstable();
    allocation_latency.sort_unstable();
    free_latency.sort_unstable();
    let median_ns = percentile(&elapsed, 0.5);
    let operations_per_second = operations.len() as f64 / (median_ns as f64 / 1_000_000_000.0);

    println!(
        concat!(
            "{{\"schema\":1,\"rmm_commit\":\"c65202f52766e12c73e293b16d6a520a069d4afc\",",
            "\"resource\":\"{}\",\"seed\":{},\"allocations\":{},\"operations\":{},",
            "\"max_allocation_bytes\":{},\"max_live_bytes\":{},\"warmups\":{},",
            "\"repetitions\":{},\"median_trace_ns\":{},\"operations_per_second\":{:.3},",
            "\"p50_operation_ns\":{},\"p99_operation_ns\":{},",
            "\"p99_allocate_ns\":{},\"p99_free_ns\":{}}}"
        ),
        resource_name,
        seed,
        allocation_count,
        operations.len(),
        maximum_allocation,
        maximum_live_bytes,
        warmups,
        repetitions,
        median_ns,
        operations_per_second,
        percentile(&latency, 0.5),
        percentile(&latency, 0.99),
        percentile(&allocation_latency, 0.99),
        percentile(&free_latency, 0.99),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let allocation_count = argument("allocations", 10_000);
    let maximum_allocation = argument("max-allocation", 1 << 20);
    let maximum_live_bytes = argument("max-live-bytes", 256 << 20);
    let warmups = argument("warmups", 2);
    let repetitions = argument("repetitions", 7);
    let seed = argument("seed", 0x5eed) as u64;
    let resource_name = resource_name();

    if allocation_count == 0
        || maximum_allocation == 0
        || maximum_live_bytes == 0
        || repetitions == 0
    {
        return Err(
            "allocations, max-allocation, max-live-bytes, and repetitions must be non-zero".into(),
        );
    }

    let context = cuda_core::CudaContext::new(0)?;
    let stream = context.default_stream();
    let operations = trace(
        allocation_count,
        maximum_allocation,
        maximum_live_bytes,
        seed,
    );

    match resource_name.as_str() {
        "cuda" => benchmark(
            &CudaMemoryResource::new(context.clone()),
            &resource_name,
            &stream,
            &operations,
            allocation_count,
            maximum_allocation,
            maximum_live_bytes,
            warmups,
            repetitions,
            seed,
        ),
        "cuda_async" => benchmark(
            &CudaAsyncMemoryResource::new(context.clone()),
            &resource_name,
            &stream,
            &operations,
            allocation_count,
            maximum_allocation,
            maximum_live_bytes,
            warmups,
            repetitions,
            seed,
        ),
        "cuda_pool" => benchmark(
            &CudaPoolMemoryResource::new(context.clone(), PoolConfig::default())?,
            &resource_name,
            &stream,
            &operations,
            allocation_count,
            maximum_allocation,
            maximum_live_bytes,
            warmups,
            repetitions,
            seed,
        ),
        "pool" => {
            let initial_size = maximum_live_bytes.min(64 << 20);
            let resource =
                PoolMemoryResource::new(CudaMemoryResource::new(context.clone()), initial_size, 0)?;
            benchmark(
                &resource,
                &resource_name,
                &stream,
                &operations,
                allocation_count,
                maximum_allocation,
                maximum_live_bytes,
                warmups,
                repetitions,
                seed,
            )
        }
        other => Err(format!("unknown resource {other:?}").into()),
    }
}
