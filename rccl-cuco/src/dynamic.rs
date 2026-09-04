use cuda_core::{
    CudaStream, DeviceBuffer as CudaDeviceBuffer, DeviceCopy,
    memory::{memcpy_dtoh_async, memcpy_htod_async},
};
use rccl_memory::{
    BorrowedCudaAsyncMemoryResource, ContiguousDeviceRange, ContiguousDeviceRangeMut,
    DeviceAllocator,
};

use crate::{
    Entries, Launch, LinearProbing, MapError, MapKey, MapValue, ProbingScheme, Sentinels,
    StaticHashMap, StaticMapConfig, backend::StaticMapBackend, entries,
};

const DEFAULT_INITIAL_CAPACITY: usize = 8;
const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.5;
const DEFAULT_GROWTH_FACTOR: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicMapConfig<K, V, P = LinearProbing, const CG: u32 = 4>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    initial_capacity: usize,
    max_load_factor: f64,
    growth_factor: f64,
    sentinels: Sentinels<K, V>,
    probing: P,
}

impl<K, V> DynamicMapConfig<K, V, LinearProbing, 4>
where
    K: MapKey,
    V: MapValue,
{
    pub fn new(sentinels: Sentinels<K, V>) -> Self {
        Self {
            initial_capacity: DEFAULT_INITIAL_CAPACITY,
            max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
            growth_factor: DEFAULT_GROWTH_FACTOR,
            sentinels,
            probing: LinearProbing::default(),
        }
    }
}

impl<K, V, P, const CG: u32> DynamicMapConfig<K, V, P, CG>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    pub const fn initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    pub const fn load_factor(mut self, max_load_factor: f64) -> Self {
        self.max_load_factor = max_load_factor;
        self
    }

    pub const fn growth_factor(mut self, growth_factor: f64) -> Self {
        self.growth_factor = growth_factor;
        self
    }

    pub fn with_probing<Q>(self, probing: Q) -> DynamicMapConfig<K, V, Q, CG>
    where
        Q: ProbingScheme<K>,
    {
        DynamicMapConfig {
            initial_capacity: self.initial_capacity,
            max_load_factor: self.max_load_factor,
            growth_factor: self.growth_factor,
            sentinels: self.sentinels,
            probing,
        }
    }

    pub fn with_cooperative_group<const NEXT_CG: u32>(self) -> DynamicMapConfig<K, V, P, NEXT_CG> {
        DynamicMapConfig {
            initial_capacity: self.initial_capacity,
            max_load_factor: self.max_load_factor,
            growth_factor: self.growth_factor,
            sentinels: self.sentinels,
            probing: self.probing,
        }
    }

    pub fn build<'stream, M>(
        self,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<
        DynamicHashMap<'stream, K, V, BorrowedCudaAsyncMemoryResource<'stream>, P, CG>,
        MapError,
    >
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let allocator = BorrowedCudaAsyncMemoryResource::new(launch.stream().context());
        self.build_in(allocator, launch)
    }

    pub fn build_in<'stream, A, M>(
        self,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<DynamicHashMap<'stream, K, V, A, P, CG>, MapError>
    where
        A: DeviceAllocator + Clone,
        M: StaticMapBackend<K, V, P, CG>,
    {
        validate_dynamic_config(self.max_load_factor, self.growth_factor, self.sentinels)?;
        let first = StaticMapConfig::with_capacity(self.initial_capacity, self.sentinels)
            .with_probing(self.probing)
            .with_cooperative_group::<CG>()
            .build_in(allocator.clone(), launch)?;
        Ok(DynamicHashMap {
            segments: vec![first],
            segment_lens: vec![0],
            len: 0,
            initial_capacity: self.initial_capacity,
            max_load_factor: self.max_load_factor,
            growth_factor: self.growth_factor,
            sentinels: self.sentinels,
            probing: self.probing,
            allocator,
        })
    }
}

pub struct DynamicHashMap<'stream, K, V, A, P, const CG: u32 = 4>
where
    K: MapKey,
    V: MapValue,
    A: DeviceAllocator,
    P: ProbingScheme<K>,
{
    segments: Vec<StaticHashMap<'stream, K, V, A, P, CG>>,
    segment_lens: Vec<usize>,
    len: usize,
    initial_capacity: usize,
    max_load_factor: f64,
    growth_factor: f64,
    sentinels: Sentinels<K, V>,
    probing: P,
    allocator: A,
}

impl<'stream, K, V, A, P, const CG: u32> DynamicHashMap<'stream, K, V, A, P, CG>
where
    K: MapKey,
    V: MapValue,
    A: DeviceAllocator + Clone,
    P: ProbingScheme<K>,
{
    pub fn insert<M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.ensure_input(&input, launch.stream())?;
        let input_len = input.len();
        if input_len == 0 {
            return Ok(0);
        }
        let before = self.len;
        self.remove_from_segments(input.keys, launch)?;
        self.ensure_insert_segment(input_len, launch)?;
        let target = self.segments.len() - 1;
        let inserted = self.segments[target].insert_or_assign(input, launch)?;
        self.segment_lens[target] = self.segment_lens[target]
            .checked_add(inserted)
            .ok_or(MapError::CapacityOverflow)?;
        self.len = self
            .len
            .checked_add(inserted)
            .ok_or(MapError::CapacityOverflow)?;
        Ok(self.len.saturating_sub(before))
    }

    pub fn get<M, KR, VO, PO>(
        &self,
        keys: &KR,
        values: &mut VO,
        present: &mut PO,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized,
        PO: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        ensure_output(keys, values, launch.stream())?;
        ensure_output(keys, present, launch.stream())?;
        if keys.is_empty() {
            return Ok(());
        }
        let count = keys.len();
        let mut merged_values = vec![self.sentinels.empty_value(); count];
        let mut merged_present = vec![0u32; count];
        let mut segment_values = CudaDeviceBuffer::<V>::zeroed(launch.stream(), count)?;
        let mut segment_present = CudaDeviceBuffer::<u32>::zeroed(launch.stream(), count)?;
        for segment in self.segments.iter().rev() {
            if merged_present.iter().all(|flag| *flag != 0) {
                break;
            }
            segment.get(keys, &mut segment_values, &mut segment_present, launch)?;
            let host_values = segment_values.to_host_vec(launch.stream())?;
            let host_present = segment_present.to_host_vec(launch.stream())?;
            for index in 0..count {
                if merged_present[index] == 0 && host_present[index] != 0 {
                    merged_values[index] = host_values[index];
                    merged_present[index] = 1;
                }
            }
        }
        write_device_prefix(&merged_values, values, launch.stream())?;
        write_device_prefix(&merged_present, present, launch.stream())?;
        Ok(())
    }

    pub fn contains_key<M, KR, OR>(
        &self,
        keys: &KR,
        output: &mut OR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        ensure_output(keys, output, launch.stream())?;
        if keys.is_empty() {
            return Ok(());
        }
        let mut values = CudaDeviceBuffer::<V>::zeroed(launch.stream(), keys.len())?;
        self.get(keys, &mut values, output, launch)
    }

    pub fn remove<M, KR>(
        &mut self,
        keys: &KR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
    {
        keys.ensure_usable_on(launch.stream())?;
        self.remove_from_segments(keys, launch)
    }

    pub fn clear<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        for segment in &mut self.segments {
            segment.clear(launch)?;
        }
        self.segment_lens.fill(0);
        self.len = 0;
        Ok(())
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.segments.iter().map(StaticHashMap::capacity).sum()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn segments(&self) -> &[StaticHashMap<'stream, K, V, A, P, CG>] {
        &self.segments
    }

    pub fn reserve<M>(
        &mut self,
        additional: usize,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        if additional != 0 {
            self.ensure_insert_segment(additional, launch)?;
        }
        Ok(())
    }

    pub fn rehash<M>(
        &mut self,
        requested_capacity: usize,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let entries_host = self.collect_entries_host(launch)?;
        let needed = capacity_for_entries(entries_host.len(), self.max_load_factor)?;
        let requested = requested_capacity.max(needed).max(self.initial_capacity);
        let mut segment = StaticMapConfig::with_capacity(requested, self.sentinels)
            .with_probing(self.probing)
            .with_cooperative_group::<CG>()
            .build_in(self.allocator.clone(), launch)?;
        if !entries_host.is_empty() {
            let (keys, values): (Vec<_>, Vec<_>) = entries_host.into_iter().unzip();
            let device_keys = CudaDeviceBuffer::from_host(launch.stream(), &keys)?;
            let device_values = CudaDeviceBuffer::from_host(launch.stream(), &values)?;
            let inserted = segment.insert(entries(&device_keys, &device_values)?, launch)?;
            if inserted != keys.len() {
                return Err(MapError::InsufficientCapacity {
                    entries: keys.len(),
                    capacity: segment.capacity(),
                });
            }
            self.len = inserted;
            self.segment_lens = vec![inserted];
        } else {
            self.len = 0;
            self.segment_lens = vec![0];
        }
        self.segments = vec![segment];
        Ok(())
    }

    fn ensure_input<KR, VR>(
        &self,
        input: &Entries<'_, K, V, KR, VR>,
        stream: &CudaStream,
    ) -> Result<(), MapError>
    where
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        input.keys.ensure_usable_on(stream)?;
        input.values.ensure_usable_on(stream)?;
        Ok(())
    }

    fn ensure_insert_segment<M>(
        &mut self,
        entries: usize,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let last = self.segments.len() - 1;
        let remaining = max_entries(self.segments[last].capacity(), self.max_load_factor)
            .saturating_sub(self.segment_lens[last]);
        if remaining >= entries {
            return Ok(());
        }
        let grown = ((self.segments[last].capacity() as f64) * self.growth_factor).ceil();
        if !grown.is_finite() || grown > usize::MAX as f64 {
            return Err(MapError::CapacityOverflow);
        }
        let requested = (grown as usize)
            .max(self.initial_capacity)
            .max(capacity_for_entries(entries, self.max_load_factor)?);
        let segment = StaticMapConfig::with_capacity(requested, self.sentinels)
            .with_probing(self.probing)
            .with_cooperative_group::<CG>()
            .build_in(self.allocator.clone(), launch)?;
        self.segments.push(segment);
        self.segment_lens.push(0);
        Ok(())
    }

    fn remove_from_segments<M, KR>(
        &mut self,
        keys: &KR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
    {
        let mut removed_total = 0usize;
        for (segment, segment_len) in self.segments.iter_mut().zip(self.segment_lens.iter_mut()) {
            let removed = segment.remove(keys, launch)?;
            *segment_len = segment_len.saturating_sub(removed);
            removed_total = removed_total
                .checked_add(removed)
                .ok_or(MapError::CapacityOverflow)?;
        }
        self.len = self.len.saturating_sub(removed_total);
        Ok(removed_total)
    }

    fn collect_entries_host<M>(
        &mut self,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Vec<(K, V)>, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let mut entries = Vec::with_capacity(self.len);
        for (segment, segment_len) in self.segments.iter_mut().zip(self.segment_lens.iter_mut()) {
            if *segment_len == 0 {
                continue;
            }
            let mut keys = CudaDeviceBuffer::<K>::zeroed(launch.stream(), segment.capacity())?;
            let mut values = CudaDeviceBuffer::<V>::zeroed(launch.stream(), segment.capacity())?;
            let count = segment.retrieve_all(&mut keys, &mut values, launch)?;
            *segment_len = count;
            let host_keys = keys.to_host_vec(launch.stream())?;
            let host_values = values.to_host_vec(launch.stream())?;
            entries.extend(host_keys.into_iter().zip(host_values).take(count));
        }
        self.len = entries.len();
        Ok(entries)
    }
}

fn validate_dynamic_config<K, V>(
    max_load_factor: f64,
    growth_factor: f64,
    sentinels: Sentinels<K, V>,
) -> Result<(), MapError>
where
    K: MapKey,
    V: MapValue,
{
    if !max_load_factor.is_finite() || max_load_factor <= 0.0 || max_load_factor > 1.0 {
        return Err(MapError::InvalidLoadFactor(max_load_factor));
    }
    if !growth_factor.is_finite() || growth_factor <= 1.0 {
        return Err(MapError::InvalidGrowthFactor(growth_factor));
    }
    if sentinels.erased_key().is_none() {
        return Err(MapError::EraseDisabled);
    }
    Ok(())
}

fn capacity_for_entries(entries: usize, max_load_factor: f64) -> Result<usize, MapError> {
    if entries == 0 {
        return Ok(0);
    }
    let requested = ((entries as f64) / max_load_factor).ceil();
    if requested > usize::MAX as f64 {
        return Err(MapError::CapacityOverflow);
    }
    Ok(requested as usize)
}

fn max_entries(capacity: usize, max_load_factor: f64) -> usize {
    ((capacity as f64) * max_load_factor).floor() as usize
}

fn ensure_output<T, O, KR, OR>(
    keys: &KR,
    output: &mut OR,
    stream: &CudaStream,
) -> Result<(), MapError>
where
    KR: ContiguousDeviceRange<T> + ?Sized,
    OR: ContiguousDeviceRangeMut<O> + ?Sized,
{
    keys.ensure_usable_on(stream)?;
    output.ensure_usable_on(stream)?;
    if output.len() < keys.len() {
        return Err(MapError::OutputTooSmall {
            needed: keys.len(),
            available: output.len(),
        });
    }
    Ok(())
}

fn write_device_prefix<T, R>(
    values: &[T],
    output: &mut R,
    stream: &CudaStream,
) -> Result<(), MapError>
where
    T: DeviceCopy,
    R: ContiguousDeviceRangeMut<T> + ?Sized,
{
    output.ensure_usable_on(stream)?;
    if output.len() < values.len() {
        return Err(MapError::OutputTooSmall {
            needed: values.len(),
            available: output.len(),
        });
    }
    let bytes = values
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(MapError::CapacityOverflow)?;
    if bytes != 0 {
        unsafe {
            memcpy_htod_async(
                output.cu_deviceptr(),
                values.as_ptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
    }
    Ok(())
}

#[allow(dead_code)]
fn read_device_range<T, R>(range: &R, stream: &CudaStream) -> Result<Vec<T>, MapError>
where
    T: DeviceCopy,
    R: ContiguousDeviceRange<T> + ?Sized,
{
    range.ensure_usable_on(stream)?;
    let mut values = Vec::with_capacity(range.len());
    let bytes = range
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(MapError::CapacityOverflow)?;
    if bytes != 0 {
        unsafe {
            memcpy_dtoh_async(
                values.as_mut_ptr(),
                range.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
            values.set_len(range.len());
        }
        stream.synchronize()?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_requires_growth_above_one() {
        let sentinels = Sentinels::new(u32::MAX, u32::MAX)
            .with_erased_key(u32::MAX - 1)
            .expect("valid sentinels");
        assert!(matches!(
            validate_dynamic_config(0.5, 1.0, sentinels),
            Err(MapError::InvalidGrowthFactor(1.0))
        ));
    }

    #[test]
    fn config_requires_erased_sentinel() {
        let sentinels = Sentinels::new(u32::MAX, u32::MAX);
        assert!(matches!(
            validate_dynamic_config(0.5, 2.0, sentinels),
            Err(MapError::EraseDisabled)
        ));
    }

    #[test]
    fn capacity_rounds_by_load_factor() {
        assert_eq!(capacity_for_entries(5, 0.5).expect("capacity"), 10);
        assert_eq!(max_entries(9, 0.5), 4);
    }
}
