//! Scan/scatter kernels for workload-facing stable compaction.

use cuda_device::cuda_module;


#[cuda_module]
#[doc(hidden)]
mod device {
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn mark_predicate<Input, I, Predicate>(
        input: Input,
        flags: DisjointSlice<I>,
        positions: DisjointSlice<I>,
        predicate: Predicate,
        len: usize,
    ) where
        Input: crate::DeviceReadAt,
        I: crate::__private::CompactionIndex,
        Predicate: Fn(Input::Item) -> bool + Copy,
    {
        let _ = launch_context;
        let mut flags = crate::__private::DeviceRegion::new(flags);
        let mut positions = crate::__private::DeviceRegion::new(positions);
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        while index < len {
            let selected = predicate(unsafe { input.read_at(index) });
            let flag = if selected { I::one() } else { I::zero() };
            flags.write(index, flag);
            positions.write(index, flag);
            if !I::GRID_STRIDE {
                break;
            }
            index += thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn mark_flagged<Flags, I>(
        selected: Flags,
        flags: DisjointSlice<I>,
        positions: DisjointSlice<I>,
        len: usize,
    ) where
        Flags: crate::DeviceReadAt<Item = u32>,
        I: crate::__private::CompactionIndex,
    {
        let _ = launch_context;
        let mut flags = crate::__private::DeviceRegion::new(flags);
        let mut positions = crate::__private::DeviceRegion::new(positions);
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        while index < len {
            let flag = if unsafe { selected.read_at(index) } != 0 {
                I::one()
            } else {
                I::zero()
            };
            flags.write(index, flag);
            positions.write(index, flag);
            if !I::GRID_STRIDE {
                break;
            }
            index += thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn mark_adjacent_unique<Input, I, Equivalent>(
        input: Input,
        flags: DisjointSlice<I>,
        positions: DisjointSlice<I>,
        equivalent: Equivalent,
        len: usize,
    ) where
        Input: crate::DeviceReadAt,
        I: crate::__private::CompactionIndex,
        Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
    {
        let _ = launch_context;
        let mut flags = crate::__private::DeviceRegion::new(flags);
        let mut positions = crate::__private::DeviceRegion::new(positions);
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        while index < len {
            let keep = index == 0
                || !equivalent(
                    unsafe { input.read_at(index - 1) },
                    unsafe { input.read_at(index) },
                );
            let flag = if keep { I::one() } else { I::zero() };
            flags.write(index, flag);
            positions.write(index, flag);
            if !I::GRID_STRIDE {
                break;
            }
            index += thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scatter_selected<Input, Output, I>(
        input: Input,
        output: Output,
        flags: &[I],
        positions: &[I],
        selected_count: DisjointSlice<u64>,
        len: usize,
    ) where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt<Item = Input::Item>,
        I: crate::__private::CompactionIndex,
    {
        let _ = launch_context;
        let total = if len == 0 {
            I::zero()
        } else {
            I::add(flags[len - 1], positions[len - 1])
        };
        if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
            let mut selected_count = crate::__private::DeviceRegion::new(selected_count);
            selected_count.write(0, total.to_u64());
        }
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        while index < len {
            if flags[index] != I::zero() {
                unsafe { output.write_at(positions[index].to_usize(), input.read_at(index)) };
            }
            if !I::GRID_STRIDE {
                break;
            }
            index += thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scatter_partition<Input, Output, I>(
        input: Input,
        output: Output,
        flags: &[I],
        positions: &[I],
        selected_count: DisjointSlice<u64>,
        len: usize,
    ) where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt<Item = Input::Item>,
        I: crate::__private::CompactionIndex,
    {
        let _ = launch_context;
        let total = if len == 0 {
            I::zero()
        } else {
            I::add(flags[len - 1], positions[len - 1])
        };
        if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
            let mut selected_count = crate::__private::DeviceRegion::new(selected_count);
            selected_count.write(0, total.to_u64());
        }
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        while index < len {
            let selected_before = positions[index].to_usize();
            let destination = if flags[index] != I::zero() {
                selected_before
            } else {
                total.to_usize() + index - selected_before
            };
            unsafe { output.write_at(destination, input.read_at(index)) };
            if !I::GRID_STRIDE {
                break;
            }
            index += thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        }
    }

    /// Reduce each adjacent-key run with one deterministic left fold.
    ///
    /// A thread is active only at a marked run head. Distinct heads own
    /// disjoint input runs and output slots, so no atomics or grid-wide
    /// synchronization are required.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn reduce_adjacent_groups<
        Keys,
        Values,
        OutputKeys,
        OutputValues,
        I,
        Operator,
    >(
        keys: Keys,
        values: Values,
        output_keys: OutputKeys,
        output_values: OutputValues,
        heads: &[I],
        positions: &[I],
        group_count: DisjointSlice<u64>,
        operator: Operator,
        len: usize,
    ) where
        Keys: crate::DeviceReadAt,
        Values: crate::DeviceReadAt,
        OutputKeys: crate::DeviceWriteAt<Item = Keys::Item>,
        OutputValues: crate::DeviceWriteAt<Item = Values::Item>,
        I: crate::__private::CompactionIndex,
        Operator: crate::__private::GroupReduceOperator<Values::Item>,
    {
        let _ = launch_context;
        let total = if len == 0 {
            I::zero()
        } else {
            I::add(heads[len - 1], positions[len - 1])
        };
        if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
            let mut group_count = crate::__private::DeviceRegion::new(group_count);
            group_count.write(0, total.to_u64());
        }

        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        while index < len {
            if heads[index] != I::zero() {
                let destination = positions[index].to_usize();
                let first_key = unsafe { keys.read_at(index) };
                let mut aggregate = unsafe { values.read_at(index) };
                let mut next = index + 1;
                while next < len && heads[next] == I::zero() {
                    aggregate = operator.combine(aggregate, unsafe { values.read_at(next) });
                    next += 1;
                }
                unsafe {
                    output_keys.write_at(destination, first_key);
                    output_values.write_at(destination, aggregate);
                }
            }
            index += stride;
        }
    }
}

#[doc(hidden)]
pub type CompactionModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<CompactionModule, cuda_host::EmbeddedModuleError> {
    unsafe { device::load(context) }
}
