use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use rccl::DeviceComparator;
use std::{error::Error, sync::Arc};

#[derive(Clone, Copy)]
struct MaskedLess {
    mask: u32,
}

unsafe impl DeviceComparator<u32> for MaskedLess {
    #[inline(always)]
    unsafe fn less(&self, left: &u32, right: &u32) -> bool {
        (left & self.mask) < (right & self.mask)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LargeKey {
    key: u32,
    payload: [u32; 127],
}

unsafe impl DeviceCopy for LargeKey {}

#[derive(Clone, Copy)]
struct LargeKeyLess;

unsafe impl DeviceComparator<LargeKey> for LargeKeyLess {
    #[inline(always)]
    unsafe fn less(&self, left: &LargeKey, right: &LargeKey) -> bool {
        left.key < right.key
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VirtualSharedKey {
    key: u32,
    payload: [u32; 255],
}

unsafe impl DeviceCopy for VirtualSharedKey {}

#[derive(Clone, Copy)]
struct VirtualSharedKeyLess;

unsafe impl DeviceComparator<VirtualSharedKey> for VirtualSharedKeyLess {
    #[inline(always)]
    unsafe fn less(&self, left: &VirtualSharedKey, right: &VirtualSharedKey) -> bool {
        left.key < right.key
    }
}

const FILL_KEYS_FIRST: u32 = 0;
const FILL_KEYS_SECOND: u32 = 1;
const FILL_DESCENDING_FIRST: u32 = 2;
const FILL_DESCENDING_SECOND: u32 = 3;
const FILL_EVENS: u32 = 4;
const FILL_ODDS: u32 = 5;

const EXPECT_KEYS_ASCENDING: u32 = 0;
const EXPECT_KEYS_DESCENDING: u32 = 1;
const EXPECT_IOTA: u32 = 2;
const EXPECT_SORTED_DESCENDING_FIRST: u32 = 3;

const EXPECT_PAIR_VALUES: u32 = 0;

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod fixtures {
    use super::*;
    use cuda_device::{
        atomic::{AtomicOrdering, DeviceAtomicU64},
        kernel, thread,
    };

    #[inline(always)]
    fn u32_fixture(index: u64, pattern: u32) -> u32 {
        match pattern {
            FILL_KEYS_FIRST => match index {
                0 => 0,
                1 => 2,
                _ => 4,
            },
            FILL_KEYS_SECOND => match index {
                0 => 0,
                1 | 2 => 3,
                _ => 4,
            },
            FILL_DESCENDING_FIRST => match index {
                0 => 9,
                1 => 7,
                2 => 5,
                _ => 1,
            },
            FILL_DESCENDING_SECOND => match index {
                0 => 8,
                1 => 6,
                2 => 5,
                _ => 2,
            },
            FILL_EVENS => index as u32 * 2,
            FILL_ODDS => index as u32 * 2 + 1,
            _ => 0,
        }
    }

    #[inline(always)]
    fn expected_u32(index: u64, pattern: u32) -> u32 {
        match pattern {
            EXPECT_KEYS_ASCENDING => match index {
                0 | 1 => 0,
                2 => 2,
                3 | 4 => 3,
                _ => 4,
            },
            EXPECT_KEYS_DESCENDING => match index {
                0 => 9,
                1 => 8,
                2 => 7,
                3 => 6,
                4 | 5 => 5,
                6 => 2,
                _ => 1,
            },
            EXPECT_IOTA => index as u32,
            EXPECT_SORTED_DESCENDING_FIRST => match index {
                0 => 1,
                1 => 5,
                2 => 7,
                _ => 9,
            },
            _ => 0,
        }
    }

    #[inline(always)]
    fn expected_pair_key(index: u64) -> i32 {
        match index {
            0 | 1 => 0,
            2 => 2,
            3 | 4 => 3,
            _ => 4,
        }
    }

    #[inline(always)]
    fn expected_u64(index: u64, pattern: u32) -> u64 {
        match pattern {
            EXPECT_PAIR_VALUES => match index {
                0 => 10,
                1 => 20,
                2 => 11,
                3 => 21,
                4 => 22,
                5 => 12,
                6 => 13,
                _ => 23,
            },
            _ => 0,
        }
    }

    #[kernel]
    pub unsafe fn fill_u32(pointer: *mut u32, len: u64, pattern: u32) {
        let index = thread::index_1d().get() as u64;
        if index < len {
            unsafe {
                pointer
                    .add(index as usize)
                    .write(u32_fixture(index, pattern))
            };
        }
    }

    #[kernel]
    pub unsafe fn fill_pair_keys(pointer: *mut i32, len: u64, second: bool) {
        let index = thread::index_1d().get() as u64;
        if index < len {
            let value = if second {
                match index {
                    0 => 0,
                    1 | 2 => 3,
                    _ => 4,
                }
            } else {
                match index {
                    0 => 0,
                    1 => 2,
                    _ => 4,
                }
            };
            unsafe { pointer.add(index as usize).write(value) };
        }
    }

    #[kernel]
    pub unsafe fn fill_u64_sequence(pointer: *mut u64, len: u64, start: u64) {
        let index = thread::index_1d().get() as u64;
        if index < len {
            unsafe { pointer.add(index as usize).write(start + index) };
        }
    }

    #[kernel]
    pub unsafe fn fill_record_words(
        pointer: *mut u32,
        records: u64,
        words_per_record: u32,
        parity: u32,
    ) {
        let index = thread::index_1d().get() as u64;
        if index < records {
            let key = index as u32 * 2 + parity;
            let record = unsafe { pointer.add(index as usize * words_per_record as usize) };
            let mut word = 0u32;
            while word < words_per_record {
                unsafe { record.add(word as usize).write(key) };
                word += 1;
            }
        }
    }

    #[kernel]
    pub unsafe fn validate_u32(pointer: *const u32, len: u64, pattern: u32, errors: *mut u64) {
        let index = thread::index_1d().get() as u64;
        if index < len
            && unsafe { pointer.add(index as usize).read() } != expected_u32(index, pattern)
        {
            unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_pair_keys(pointer: *const i32, len: u64, errors: *mut u64) {
        let index = thread::index_1d().get() as u64;
        if index < len && unsafe { pointer.add(index as usize).read() } != expected_pair_key(index)
        {
            unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_u64(pointer: *const u64, len: u64, pattern: u32, errors: *mut u64) {
        let index = thread::index_1d().get() as u64;
        if index < len
            && unsafe { pointer.add(index as usize).read() } != expected_u64(index, pattern)
        {
            unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_record_words(
        pointer: *const u32,
        records: u64,
        words_per_record: u32,
        errors: *mut u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index < records {
            let record = unsafe { pointer.add(index as usize * words_per_record as usize) };
            let expected = index as u32;
            let mut word = 0u32;
            let mut valid = true;
            while word < words_per_record {
                valid &= unsafe { record.add(word as usize).read() } == expected;
                word += 1;
            }
            if !valid {
                unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

fn launch_for(len: usize) -> Result<LaunchConfig, Box<dyn Error>> {
    Ok(LaunchConfig::for_num_elems(u32::try_from(len)?))
}

fn fill_u32(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &mut DeviceBuffer<u32>,
    pattern: u32,
) -> Result<(), Box<dyn Error>> {
    if !buffer.is_empty() {
        unsafe {
            module.fill_u32(
                stream,
                launch_for(buffer.len())?,
                buffer.cu_deviceptr() as *mut u32,
                buffer.len() as u64,
                pattern,
            )?;
        }
    }
    Ok(())
}

fn fill_pair_keys(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &mut DeviceBuffer<i32>,
    second: bool,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        module.fill_pair_keys(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *mut i32,
            buffer.len() as u64,
            second,
        )?;
    }
    Ok(())
}

fn fill_u64_sequence(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &mut DeviceBuffer<u64>,
    start: u64,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        module.fill_u64_sequence(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *mut u64,
            buffer.len() as u64,
            start,
        )?;
    }
    Ok(())
}

fn fill_record_words<T: DeviceCopy>(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &mut DeviceBuffer<T>,
    words_per_record: u32,
    parity: u32,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        module.fill_record_words(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *mut u32,
            buffer.len() as u64,
            words_per_record,
            parity,
        )?;
    }
    Ok(())
}

fn check_status(
    stream: &Arc<CudaStream>,
    status: &DeviceBuffer<u64>,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let mut errors = 0u64;
    status.copy_to_host(stream, std::slice::from_mut(&mut errors))?;
    if errors != 0 {
        return Err(format!("{label} found {errors} incorrect device elements").into());
    }
    Ok(())
}

fn validate_u32(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<u32>,
    pattern: u32,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let status = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    unsafe {
        module.validate_u32(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *const u32,
            buffer.len() as u64,
            pattern,
            status.cu_deviceptr() as *mut u64,
        )?;
    }
    check_status(stream, &status, label)
}

fn validate_pair_keys(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<i32>,
) -> Result<(), Box<dyn Error>> {
    let status = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    unsafe {
        module.validate_pair_keys(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *const i32,
            buffer.len() as u64,
            status.cu_deviceptr() as *mut u64,
        )?;
    }
    check_status(stream, &status, "merge_by_key keys")
}

fn validate_u64(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<u64>,
    pattern: u32,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let status = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    unsafe {
        module.validate_u64(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *const u64,
            buffer.len() as u64,
            pattern,
            status.cu_deviceptr() as *mut u64,
        )?;
    }
    check_status(stream, &status, label)
}

fn validate_record_words<T: DeviceCopy>(
    module: &fixtures::LoadedModule,
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<T>,
    words_per_record: u32,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let status = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    unsafe {
        module.validate_record_words(
            stream,
            launch_for(buffer.len())?,
            buffer.cu_deviceptr() as *const u32,
            buffer.len() as u64,
            words_per_record,
            status.cu_deviceptr() as *mut u64,
        )?;
    }
    check_status(stream, &status, label)
}

fn test_sort_with_shared_execution(
    exec: &rccl::cuda::Execution<'_>,
    stream: &Arc<CudaStream>,
    module: &fixtures::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let mut keys = DeviceBuffer::<u32>::zeroed(stream, 4)?;
    fill_u32(module, stream, &mut keys, FILL_DESCENDING_FIRST)?;
    rccl::sort(exec, &mut keys)?;
    validate_u32(
        module,
        stream,
        &keys,
        EXPECT_SORTED_DESCENDING_FIRST,
        "shared execution sort",
    )
}

fn test_keys(
    exec: &rccl::cuda::Execution<'_>,
    stream: &Arc<CudaStream>,
    module: &fixtures::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let mut first = DeviceBuffer::<u32>::zeroed(stream, 4)?;
    let mut second = DeviceBuffer::<u32>::zeroed(stream, 4)?;
    fill_u32(module, stream, &mut first, FILL_KEYS_FIRST)?;
    fill_u32(module, stream, &mut second, FILL_KEYS_SECOND)?;
    let mut output = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    let end = rccl::merge(exec, &first, &second, &mut output)?;
    assert_eq!(end, output.len());
    validate_u32(
        module,
        stream,
        &output,
        EXPECT_KEYS_ASCENDING,
        "ascending merge",
    )?;

    fill_u32(module, stream, &mut first, FILL_DESCENDING_FIRST)?;
    fill_u32(module, stream, &mut second, FILL_DESCENDING_SECOND)?;
    rccl::merge_by(exec, &first, &second, &mut output, rccl::Greater)?;
    validate_u32(
        module,
        stream,
        &output,
        EXPECT_KEYS_DESCENDING,
        "descending merge",
    )?;

    let empty = DeviceBuffer::<u32>::zeroed(stream, 0)?;
    let mut empty_output = DeviceBuffer::<u32>::zeroed(stream, 0)?;
    assert_eq!(rccl::merge(exec, &empty, &empty, &mut empty_output)?, 0);

    fill_u32(module, stream, &mut first, FILL_EVENS)?;
    fill_u32(module, stream, &mut second, FILL_ODDS)?;
    rccl::merge_by(
        exec,
        &first,
        &second,
        &mut output,
        MaskedLess { mask: u32::MAX },
    )?;
    validate_u32(module, stream, &output, EXPECT_IOTA, "stateful merge_by")?;
    Ok(())
}

fn test_pairs(
    exec: &rccl::cuda::Execution<'_>,
    stream: &Arc<CudaStream>,
    module: &fixtures::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let mut first_keys = DeviceBuffer::<i32>::zeroed(stream, 4)?;
    let mut first_values = DeviceBuffer::<u64>::zeroed(stream, 4)?;
    let mut second_keys = DeviceBuffer::<i32>::zeroed(stream, 4)?;
    let mut second_values = DeviceBuffer::<u64>::zeroed(stream, 4)?;
    fill_pair_keys(module, stream, &mut first_keys, false)?;
    fill_u64_sequence(module, stream, &mut first_values, 10)?;
    fill_pair_keys(module, stream, &mut second_keys, true)?;
    fill_u64_sequence(module, stream, &mut second_values, 20)?;
    let mut output_keys = DeviceBuffer::<i32>::zeroed(stream, 8)?;
    let mut output_values = DeviceBuffer::<u64>::zeroed(stream, 8)?;
    let ends = rccl::merge_by_key(
        exec,
        &first_keys,
        &first_values,
        &second_keys,
        &second_values,
        &mut output_keys,
        &mut output_values,
    )?;
    assert_eq!(ends, (8, 8));
    validate_pair_keys(module, stream, &output_keys)?;
    validate_u64(
        module,
        stream,
        &output_values,
        EXPECT_PAIR_VALUES,
        "merge_by_key stable values",
    )?;
    Ok(())
}

fn test_fallback(
    exec: &rccl::cuda::Execution<'_>,
    stream: &Arc<CudaStream>,
    module: &fixtures::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let mut first = DeviceBuffer::<LargeKey>::zeroed(stream, 3)?;
    let mut second = DeviceBuffer::<LargeKey>::zeroed(stream, 3)?;
    fill_record_words(module, stream, &mut first, 128, 0)?;
    fill_record_words(module, stream, &mut second, 128, 1)?;
    let mut output = DeviceBuffer::<LargeKey>::zeroed(stream, 6)?;
    rccl::merge_by(exec, &first, &second, &mut output, LargeKeyLess)?;
    validate_record_words(module, stream, &output, 128, "large-record fallback")?;

    // A 64-thread fallback tile still exceeds the portable shared-memory
    // ceiling for this record. CUB's next step restores the normal agent shape
    // without load-to-shared and gives it global-backed virtual shared memory.
    let mut first = DeviceBuffer::<VirtualSharedKey>::zeroed(stream, 2)?;
    let mut second = DeviceBuffer::<VirtualSharedKey>::zeroed(stream, 2)?;
    fill_record_words(module, stream, &mut first, 256, 0)?;
    fill_record_words(module, stream, &mut second, 256, 1)?;
    let mut output = DeviceBuffer::<VirtualSharedKey>::zeroed(stream, 4)?;
    rccl::merge_by(exec, &first, &second, &mut output, VirtualSharedKeyLess)?;
    validate_record_words(module, stream, &output, 256, "virtual-shared fallback")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let exec = rccl::cuda::par(&stream)?;
    let module = fixtures::load(&context)?;

    eprintln!("shared execution: ordinary Thrust sort");
    test_sort_with_shared_execution(&exec, &stream, &module)?;
    eprintln!("merge matrix: keys");
    test_keys(&exec, &stream, &module)?;
    eprintln!("merge matrix: pairs");
    test_pairs(&exec, &stream, &module)?;
    eprintln!("merge matrix: large/vsmem");
    test_fallback(&exec, &stream, &module)?;
    stream.synchronize()?;
    println!("shared Execution sort + CCCL contiguous merge matrix: PASS (4 suites)");
    Ok(())
}
