//! Fixed CUDA entry boundary for set operations.

use cuda_device::cuda_module;

#[cuda_module]
#[doc(hidden)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA entry points preserve the pinned CUB launch ABI"
)]
mod device {
    use cuda_device::{DisjointSlice, Uniform, kernel, launch_bounds, launch_contract, thread};

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub fn device_set_init<I>(tile_state: DisjointSlice<I::Transaction>, num_tiles: Uniform<usize>)
    where
        I: crate::__private::SetIndex,
    {
        crate::__private::set_init_agent::<I>(tile_state, num_tiles.get());
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn device_set_partition<I, K, F, const ITEMS_PER_TILE: usize>(
        first: &[K],
        second: &[K],
        order: F,
        partitions: DisjointSlice<I::Transaction>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
    {
        crate::__private::set_partition_agent::<I, K, F, _, _, ITEMS_PER_TILE>(
            first,
            second,
            order,
            partitions,
            launch_context,
        );
    }
    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn device_set_arithmetic_partition<I, K, F, const ITEMS_PER_TILE: usize>(
        first_start: Uniform<K>,
        first_step: Uniform<K>,
        first_len: Uniform<usize>,
        second_start: Uniform<K>,
        second_step: Uniform<K>,
        second_len: Uniform<usize>,
        order: F,
        partitions: DisjointSlice<I::Transaction>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy
            + ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = K>
            + ::core::ops::Mul<Output = K>,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        let first = crate::__private::ArithmeticSetRange::new(
            first_start.get(),
            first_step.get(),
            first_len.get(),
        );
        let second = crate::__private::ArithmeticSetRange::new(
            second_start.get(),
            second_step.get(),
            second_len.get(),
        );
        crate::__private::set_partition_agent::<I, K, F, _, _, ITEMS_PER_TILE>(
            first,
            second,
            order,
            partitions,
            launch_context,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_set_keys<I, K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first: &[K],
        second: &[K],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output: DisjointSlice<K>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::new(SHARED_BYTES);
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_keys_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state, output,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1))]
    pub fn device_set_keys_virtual<I, K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first: &[K],
        second: &[K],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output: DisjointSlice<K>,
        virtual_shared: DisjointSlice<u8>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::from_virtual_block(
            virtual_shared,
            thread::blockIdx_x(),
            SHARED_BYTES,
        );
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_keys_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state, output,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_set_discard<I, K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first: &[K],
        second: &[K],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::new(SHARED_BYTES);
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_keys_discard_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1))]
    pub fn device_set_discard_virtual<I, K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first: &[K],
        second: &[K],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        virtual_shared: DisjointSlice<u8>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::from_virtual_block(
            virtual_shared,
            thread::blockIdx_x(),
            SHARED_BYTES,
        );
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_keys_discard_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_set_arithmetic<I, K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first_start: Uniform<K>,
        first_step: Uniform<K>,
        first_len: Uniform<usize>,
        second_start: Uniform<K>,
        second_step: Uniform<K>,
        second_len: Uniform<usize>,
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output: DisjointSlice<K>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy
            + ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = K>
            + ::core::ops::Mul<Output = K>,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::new(SHARED_BYTES);
        let first = crate::__private::ArithmeticSetRange::new(
            first_start.get(),
            first_step.get(),
            first_len.get(),
        );
        let second = crate::__private::ArithmeticSetRange::new(
            second_start.get(),
            second_step.get(),
            second_len.get(),
        );
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_arithmetic_keys_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state, output,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1))]
    pub fn device_set_arithmetic_virtual<
        I,
        K,
        F,
        Op,
        const ITEMS: usize,
        const SHARED_BYTES: usize,
    >(
        first_start: Uniform<K>,
        first_step: Uniform<K>,
        first_len: Uniform<usize>,
        second_start: Uniform<K>,
        second_step: Uniform<K>,
        second_len: Uniform<usize>,
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output: DisjointSlice<K>,
        virtual_shared: DisjointSlice<u8>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy
            + ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = K>
            + ::core::ops::Mul<Output = K>,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::from_virtual_block(
            virtual_shared,
            thread::blockIdx_x(),
            SHARED_BYTES,
        );
        let first = crate::__private::ArithmeticSetRange::new(
            first_start.get(),
            first_step.get(),
            first_len.get(),
        );
        let second = crate::__private::ArithmeticSetRange::new(
            second_start.get(),
            second_step.get(),
            second_len.get(),
        );
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        crate::__private::set_arithmetic_keys_agent::<K, I, F, Op, 512, ITEMS>(
            first, second, order, state, output,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_set_pairs<I, K, V, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first_keys: &[K],
        second_keys: &[K],
        first_values: &[V],
        second_values: &[V],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output_keys: DisjointSlice<K>,
        output_values: DisjointSlice<V>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::new(SHARED_BYTES);
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        let output = crate::__private::SetPairOutput::new(output_keys, output_values);
        crate::__private::set_pairs_agent::<K, V, I, F, Op, 512, ITEMS>(
            first_keys,
            second_keys,
            first_values,
            second_values,
            order,
            state,
            output,
        );
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1))]
    pub fn device_set_pairs_virtual<I, K, V, F, Op, const ITEMS: usize, const SHARED_BYTES: usize>(
        first_keys: &[K],
        second_keys: &[K],
        first_values: &[V],
        second_values: &[V],
        partitions: &[I::Transaction],
        order: F,
        tile_state: DisjointSlice<I::Transaction>,
        output_count: DisjointSlice<usize>,
        output_keys: DisjointSlice<K>,
        output_values: DisjointSlice<V>,
        virtual_shared: DisjointSlice<u8>,
    ) where
        I: crate::__private::SetIndex,
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
    {
        let shared = crate::__private::BlockSharedMemory::from_virtual_block(
            virtual_shared,
            thread::blockIdx_x(),
            SHARED_BYTES,
        );
        let state =
            crate::__private::SetAgentState::new(partitions, tile_state, output_count, shared);
        let output = crate::__private::SetPairOutput::new(output_keys, output_values);
        crate::__private::set_pairs_agent::<K, V, I, F, Op, 512, ITEMS>(
            first_keys,
            second_keys,
            first_values,
            second_values,
            order,
            state,
            output,
        );
    }
}

#[doc(hidden)]
pub type SetModule = device::LoadedModule;

impl<I, K, F, Op> crate::__private::SetKernel<I, K, F, Op> for SetModule
where
    I: crate::__private::SetIndex,
    K: crate::__private::cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    Op: crate::__private::SerialSetOperation,
{
    fn launch_set_keys<
        First,
        Second,
        O,
        D,
        P,
        W,
        Output,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first: &First,
        second: &Second,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        First: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Second: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        O: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        D: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W: crate::__private::cuda_host::KernelSliceArgMut<Elem = u8>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let init_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.div_ceil(128).min(u32::MAX as usize) as u32,
            128,
            0,
        );
        let prepared = self
            .prepare_device_set_init::<I>(init_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_init::<I>(stream, &prepared, tile_state, num_tiles)
            .map_err(crate::DeviceAlgorithmError::Launch)?;

        let partition_config = crate::__private::cuda_core::LaunchConfig1D::new(
            (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let prepared = self
            .prepare_device_set_partition::<I, K, F, TILE_ITEMS>(partition_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_partition::<I, K, F, TILE_ITEMS>(
            stream, &prepared, first, second, order, partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;

        let virtualized = !crate::__private::cuda_host::KernelSliceArg::is_empty(&*virtual_shared);
        let agent_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.min(u32::MAX as usize) as u32,
            512,
            if virtualized { 0 } else { SHARED as u32 },
        );
        if virtualized {
            let prepared = self
                .prepare_device_set_keys_virtual::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_keys_virtual::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first,
                second,
                &*partitions,
                order,
                tile_state,
                output_count,
                output,
                virtual_shared,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        } else {
            let prepared = self
                .prepare_device_set_keys::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_keys::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first,
                second,
                &*partitions,
                order,
                tile_state,
                output_count,
                output,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        }
    }

    fn launch_set_discard<
        First,
        Second,
        O,
        D,
        P,
        W,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first: &First,
        second: &Second,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        First: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Second: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        O: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        D: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W: crate::__private::cuda_host::KernelSliceArgMut<Elem = u8>,
    {
        let init_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.div_ceil(128).min(u32::MAX as usize) as u32,
            128,
            0,
        );
        let prepared = self
            .prepare_device_set_init::<I>(init_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_init::<I>(stream, &prepared, tile_state, num_tiles)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        let partition_config = crate::__private::cuda_core::LaunchConfig1D::new(
            (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let prepared = self
            .prepare_device_set_partition::<I, K, F, TILE_ITEMS>(partition_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_partition::<I, K, F, TILE_ITEMS>(
            stream, &prepared, first, second, order, partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;
        let virtualized = !crate::__private::cuda_host::KernelSliceArg::is_empty(&*virtual_shared);
        let agent_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.min(u32::MAX as usize) as u32,
            512,
            if virtualized { 0 } else { SHARED as u32 },
        );
        if virtualized {
            let prepared = self
                .prepare_device_set_discard_virtual::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_discard_virtual::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first,
                second,
                &*partitions,
                order,
                tile_state,
                output_count,
                virtual_shared,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        } else {
            let prepared = self
                .prepare_device_set_discard::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_discard::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first,
                second,
                &*partitions,
                order,
                tile_state,
                output_count,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        }
    }

    fn launch_set_pairs<
        V,
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        O,
        D,
        P,
        W,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first_keys: &FirstKeys,
        first_values: &FirstValues,
        second_keys: &SecondKeys,
        second_values: &SecondValues,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        V: crate::__private::cuda_core::DeviceCopy,
        FirstKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        FirstValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        SecondKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        SecondValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        O: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        D: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W: crate::__private::cuda_host::KernelSliceArgMut<Elem = u8>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let init_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.div_ceil(128).min(u32::MAX as usize) as u32,
            128,
            0,
        );
        let prepared = self
            .prepare_device_set_init::<I>(init_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_init::<I>(stream, &prepared, tile_state, num_tiles)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        let partition_config = crate::__private::cuda_core::LaunchConfig1D::new(
            (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let prepared = self
            .prepare_device_set_partition::<I, K, F, TILE_ITEMS>(partition_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_partition::<I, K, F, TILE_ITEMS>(
            stream,
            &prepared,
            first_keys,
            second_keys,
            order,
            partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;
        let virtualized = !crate::__private::cuda_host::KernelSliceArg::is_empty(&*virtual_shared);
        let agent_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.min(u32::MAX as usize) as u32,
            512,
            if virtualized { 0 } else { SHARED as u32 },
        );
        if virtualized {
            let prepared = self
                .prepare_device_set_pairs_virtual::<I, K, V, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_pairs_virtual::<I, K, V, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first_keys,
                second_keys,
                first_values,
                second_values,
                &*partitions,
                order,
                tile_state,
                output_count,
                output_keys,
                output_values,
                virtual_shared,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        } else {
            let prepared = self
                .prepare_device_set_pairs::<I, K, V, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_pairs::<I, K, V, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first_keys,
                second_keys,
                first_values,
                second_values,
                &*partitions,
                order,
                tile_state,
                output_count,
                output_keys,
                output_values,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        }
    }
}

impl<I, K, F, Op> crate::__private::ArithmeticSetKernel<I, K, F, Op> for SetModule
where
    I: crate::__private::SetIndex,
    K: crate::__private::cuda_core::DeviceCopy
        + ::core::ops::Add<Output = K>
        + ::core::ops::Mul<Output = K>
        + ::core::convert::TryFrom<usize>,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    Op: crate::__private::SerialSetOperation,
{
    fn launch_set_arithmetic<
        O,
        D,
        P,
        W,
        Output,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first_start: K,
        first_step: K,
        first_len: usize,
        second_start: K,
        second_step: K,
        second_len: usize,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        O: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        D: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P: crate::__private::cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W: crate::__private::cuda_host::KernelSliceArgMut<Elem = u8>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let init_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.div_ceil(128).min(u32::MAX as usize) as u32,
            128,
            0,
        );
        let prepared = self
            .prepare_device_set_init::<I>(init_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_init::<I>(stream, &prepared, tile_state, num_tiles)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        let partition_config = crate::__private::cuda_core::LaunchConfig1D::new(
            (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let prepared = self
            .prepare_device_set_arithmetic_partition::<I, K, F, TILE_ITEMS>(partition_config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_set_arithmetic_partition::<I, K, F, TILE_ITEMS>(
            stream,
            &prepared,
            first_start,
            first_step,
            first_len,
            second_start,
            second_step,
            second_len,
            order,
            partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;
        let virtualized = !crate::__private::cuda_host::KernelSliceArg::is_empty(&*virtual_shared);
        let agent_config = crate::__private::cuda_core::LaunchConfig1D::new(
            num_tiles.min(u32::MAX as usize) as u32,
            512,
            if virtualized { 0 } else { SHARED as u32 },
        );
        if virtualized {
            let prepared = self
                .prepare_device_set_arithmetic_virtual::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_arithmetic_virtual::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first_start,
                first_step,
                first_len,
                second_start,
                second_step,
                second_len,
                &*partitions,
                order,
                tile_state,
                output_count,
                output,
                virtual_shared,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        } else {
            let prepared = self
                .prepare_device_set_arithmetic::<I, K, F, Op, ITEMS, SHARED>(agent_config)
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.device_set_arithmetic::<I, K, F, Op, ITEMS, SHARED>(
                stream,
                &prepared,
                first_start,
                first_step,
                first_len,
                second_start,
                second_step,
                second_len,
                &*partitions,
                order,
                tile_state,
                output_count,
                output,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        }
    }
}

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<SetModule, cuda_host::EmbeddedModuleError> {
    // The module and loader are defined together, so every host method is
    // tied to the generic Rust entry point above.
    unsafe { device::load(context) }
}
