use core::{
    fmt,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use cuda_core::{DeviceBuffer, DeviceCopy};
pub use rccl_cub::{ArithmeticRange, arithmetic_range};
use rccl_cub::{ContiguousRange, DeviceRange, RandomAccessRange, ReadRange, WriteRange};

/// Invalid construction of a finite device-range expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeError {
    LengthMismatch {
        first: usize,
        second: usize,
    },
    ZeroStep,
    OutOfBounds {
        start: usize,
        end: usize,
        len: usize,
    },
    BoundOverflow,
}

impl fmt::Display for RangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { first, second } => {
                write!(
                    formatter,
                    "device range lengths differ: {first} and {second}"
                )
            }
            Self::ZeroStep => formatter.write_str("device range step must be non-zero"),
            Self::OutOfBounds { start, end, len } => write!(
                formatter,
                "device subrange {start}..{end} is outside range length {len}"
            ),
            Self::BoundOverflow => formatter.write_str("device range bound overflowed"),
        }
    }
}

impl std::error::Error for RangeError {}

pub fn repeat_n<T>(value: T, len: usize) -> RepeatN<T> {
    RepeatN::new(value, len)
}

pub fn discard<T>(len: usize) -> Discard<T> {
    Discard::new(len)
}

pub fn indexed_output<F, T>(len: usize, write: F) -> IndexedOutput<F, T> {
    IndexedOutput::new(len, write)
}

/// A shared borrow of one complete device buffer.
///
/// This is the device counterpart of a slice iterator's borrowed source: it
/// may be copied and composed freely, but it cannot outlive the allocation it
/// reads.  Algorithms consume the view expression rather than accepting an
/// unrelated pair of cursors.
///
/// ```compile_fail
/// use cuda_core::{DeviceBuffer, DeviceCopy};
/// use rccl_thrust::{DeviceBufferExt, DeviceView};
///
/// fn escape<T: DeviceCopy>(buffer: &DeviceBuffer<T>) -> DeviceView<'static, T> {
///     buffer.view()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct DeviceView<'a, T: DeviceCopy> {
    buffer: &'a DeviceBuffer<T>,
}

impl<'a, T: DeviceCopy> DeviceView<'a, T> {
    fn new(buffer: &'a DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    #[doc(hidden)]
    pub fn __into_buffer(self) -> &'a DeviceBuffer<T> {
        self.buffer
    }

    /// Combine two equally-sized borrowed buffers as one structure-of-arrays
    /// sequence.
    pub fn with_values<V: DeviceCopy>(
        self,
        values: DeviceView<'a, V>,
    ) -> Result<KeyValueRange<'a, T, V>, RangeError> {
        KeyValueRange::new(self, values)
    }
}

impl<T: DeviceCopy> DeviceRange for DeviceView<'_, T> {
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<T: DeviceCopy> ReadRange for DeviceView<'_, T> {
    type Item = T;
}

impl<T: DeviceCopy> RandomAccessRange for DeviceView<'_, T> {}
impl<T: DeviceCopy> ContiguousRange for DeviceView<'_, T> {}

/// An exclusive borrow of one complete device buffer.
///
/// Unlike a C++ output iterator this value is affine: moving it transfers the
/// exclusive borrow, and no second writer can be constructed while it lives.
///
/// ```compile_fail
/// use cuda_core::{DeviceBuffer, DeviceCopy};
/// use rccl_thrust::DeviceBufferExt;
///
/// fn alias<T: DeviceCopy>(buffer: &mut DeviceBuffer<T>) {
///     let first = buffer.view_mut();
///     let second = buffer.view_mut();
///     drop((first, second));
/// }
/// ```
pub struct DeviceViewMut<'a, T: DeviceCopy> {
    buffer: &'a mut DeviceBuffer<T>,
}

impl<'a, T: DeviceCopy> DeviceViewMut<'a, T> {
    fn new(buffer: &'a mut DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    pub fn as_view(&self) -> DeviceView<'_, T> {
        DeviceView::new(self.buffer)
    }

    pub(crate) fn into_buffer(self) -> &'a mut DeviceBuffer<T> {
        self.buffer
    }

    /// Combine two equally-sized exclusive borrows as one writable
    /// structure-of-arrays sequence.
    pub fn with_values<V: DeviceCopy>(
        self,
        values: DeviceViewMut<'a, V>,
    ) -> Result<KeyValueRangeMut<'a, T, V>, RangeError> {
        KeyValueRangeMut::new(self, values)
    }
}

impl<T: DeviceCopy> DeviceRange for DeviceViewMut<'_, T> {
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<T: DeviceCopy> ReadRange for DeviceViewMut<'_, T> {
    type Item = T;
}

impl<T: DeviceCopy> WriteRange for DeviceViewMut<'_, T> {
    type Item = T;
}

impl<T: DeviceCopy> RandomAccessRange for DeviceViewMut<'_, T> {}
impl<T: DeviceCopy> ContiguousRange for DeviceViewMut<'_, T> {}

/// Borrowing entry points for device-sequence composition.
pub trait DeviceBufferExt<T: DeviceCopy> {
    fn view(&self) -> DeviceView<'_, T>;

    fn view_mut(&mut self) -> DeviceViewMut<'_, T>;
}

impl<T: DeviceCopy> DeviceBufferExt<T> for DeviceBuffer<T> {
    fn view(&self) -> DeviceView<'_, T> {
        DeviceView::new(self)
    }

    fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::new(self)
    }
}

/// A finite constant device-range expression.
#[derive(Clone, Copy, Debug)]
pub struct RepeatN<T> {
    value: T,
    len: usize,
}

impl<T> RepeatN<T> {
    pub fn new(value: T, len: usize) -> Self {
        Self { value, len }
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> DeviceRange for RepeatN<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: DeviceCopy> ReadRange for RepeatN<T> {
    type Item = T;
}

impl<T> RandomAccessRange for RepeatN<T> {}

/// An output expression that intentionally discards device writes.
#[derive(Clone, Copy, Debug)]
pub struct Discard<T> {
    len: usize,
    item: PhantomData<fn(T)>,
}

impl<T> Discard<T> {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            item: PhantomData,
        }
    }
}

impl<T> DeviceRange for Discard<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: DeviceCopy> WriteRange for Discard<T> {
    type Item = T;
}

impl<T> RandomAccessRange for Discard<T> {}

/// A borrowed structure-of-arrays key/value input range.
#[derive(Clone, Copy)]
pub struct KeyValueRange<'a, K: DeviceCopy, V: DeviceCopy> {
    keys: &'a DeviceBuffer<K>,
    values: &'a DeviceBuffer<V>,
}

impl<'a, K: DeviceCopy, V: DeviceCopy> KeyValueRange<'a, K, V> {
    pub fn new(keys: DeviceView<'a, K>, values: DeviceView<'a, V>) -> Result<Self, RangeError> {
        let keys = keys.__into_buffer();
        let values = values.__into_buffer();
        if keys.len() != values.len() {
            return Err(RangeError::LengthMismatch {
                first: keys.len(),
                second: values.len(),
            });
        }
        Ok(Self { keys, values })
    }

    pub fn parts(self) -> (&'a DeviceBuffer<K>, &'a DeviceBuffer<V>) {
        (self.keys, self.values)
    }
}

impl<K: DeviceCopy, V: DeviceCopy> DeviceRange for KeyValueRange<'_, K, V> {
    fn len(&self) -> usize {
        self.keys.len()
    }
}

impl<K: DeviceCopy, V: DeviceCopy> ReadRange for KeyValueRange<'_, K, V>
where
    (K, V): DeviceCopy,
{
    type Item = (K, V);
}

impl<K: DeviceCopy, V: DeviceCopy> RandomAccessRange for KeyValueRange<'_, K, V> {}

/// A borrowed structure-of-arrays key/value range with exclusive output
/// ownership of both component buffers.
pub struct KeyValueRangeMut<'a, K: DeviceCopy, V: DeviceCopy> {
    keys: &'a mut DeviceBuffer<K>,
    values: &'a mut DeviceBuffer<V>,
}

impl<'a, K: DeviceCopy, V: DeviceCopy> KeyValueRangeMut<'a, K, V> {
    pub fn new(
        keys: DeviceViewMut<'a, K>,
        values: DeviceViewMut<'a, V>,
    ) -> Result<Self, RangeError> {
        let keys = keys.into_buffer();
        let values = values.into_buffer();
        if keys.len() != values.len() {
            return Err(RangeError::LengthMismatch {
                first: keys.len(),
                second: values.len(),
            });
        }
        Ok(Self { keys, values })
    }

    pub fn parts_mut(&mut self) -> (&mut DeviceBuffer<K>, &mut DeviceBuffer<V>) {
        (&mut *self.keys, &mut *self.values)
    }

    /// Create a shorter exclusive view for one synchronous operation.
    pub fn reborrow(&mut self) -> KeyValueRangeMut<'_, K, V> {
        KeyValueRangeMut {
            keys: &mut *self.keys,
            values: &mut *self.values,
        }
    }

    pub fn into_parts(self) -> (&'a mut DeviceBuffer<K>, &'a mut DeviceBuffer<V>) {
        (self.keys, self.values)
    }
}

impl<K: DeviceCopy, V: DeviceCopy> DeviceRange for KeyValueRangeMut<'_, K, V> {
    fn len(&self) -> usize {
        self.keys.len()
    }
}

impl<K: DeviceCopy, V: DeviceCopy> ReadRange for KeyValueRangeMut<'_, K, V>
where
    (K, V): DeviceCopy,
{
    type Item = (K, V);
}

impl<K: DeviceCopy, V: DeviceCopy> WriteRange for KeyValueRangeMut<'_, K, V>
where
    (K, V): DeviceCopy,
{
    type Item = (K, V);
}

impl<K: DeviceCopy, V: DeviceCopy> RandomAccessRange for KeyValueRangeMut<'_, K, V> {}

/// A checked contiguous logical subrange expression.
#[derive(Clone, Copy, Debug)]
pub struct Subrange<R> {
    inner: R,
    start: usize,
    len: usize,
}

impl<R: DeviceRange> Subrange<R> {
    pub fn new(inner: R, start: usize, len: usize) -> Option<Self> {
        let end = start.checked_add(len)?;
        (end <= inner.len()).then_some(Self { inner, start, len })
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> DeviceRange for Subrange<R> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<R: ReadRange> ReadRange for Subrange<R> {
    type Item = R::Item;
}

impl<R: WriteRange> WriteRange for Subrange<R> {
    type Item = R::Item;
}

impl<R: RandomAccessRange> RandomAccessRange for Subrange<R> {}
impl<R: ContiguousRange> ContiguousRange for Subrange<R> {}

/// A reversed logical range expression.
#[derive(Clone, Copy, Debug)]
pub struct Rev<R> {
    inner: R,
    len: usize,
}

impl<R: DeviceRange> Rev<R> {
    pub fn new(inner: R) -> Self {
        let len = inner.len();
        Self { inner, len }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> DeviceRange for Rev<R> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<R: ReadRange> ReadRange for Rev<R> {
    type Item = R::Item;
}

impl<R: WriteRange> WriteRange for Rev<R> {
    type Item = R::Item;
}

impl<R: RandomAccessRange> RandomAccessRange for Rev<R> {}

/// A range expression that keeps every `step`th position.
#[derive(Clone, Copy, Debug)]
pub struct StepBy<R> {
    inner: R,
    step: usize,
    len: usize,
}

impl<R: DeviceRange> StepBy<R> {
    pub fn new(inner: R, step: usize) -> Option<Self> {
        if step == 0 {
            return None;
        }
        let len = inner.len().div_ceil(step);
        Some(Self { inner, step, len })
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> DeviceRange for StepBy<R> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<R: ReadRange> ReadRange for StepBy<R> {
    type Item = R::Item;
}

impl<R: WriteRange> WriteRange for StepBy<R> {
    type Item = R::Item;
}

impl<R: RandomAccessRange> RandomAccessRange for StepBy<R> {}

/// A zero-copy mapped input expression. Construction does not evaluate the
/// closure; a generated algorithm backend compiles it into device code.
#[derive(Clone, Copy, Debug)]
pub struct Map<R, F, T> {
    inner: R,
    map: F,
    output: PhantomData<fn() -> T>,
}

impl<R, F, T> Map<R, F, T> {
    pub fn new(inner: R, map: F) -> Self {
        Self {
            inner,
            map,
            output: PhantomData,
        }
    }

    pub fn parts(&self) -> (&R, &F) {
        (&self.inner, &self.map)
    }
}

impl<R: DeviceRange, F, T> DeviceRange for Map<R, F, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<R, F, T> ReadRange for Map<R, F, T>
where
    R: ReadRange,
    F: Fn(R::Item) -> T,
    T: DeviceCopy,
{
    type Item = T;
}

impl<R: RandomAccessRange, F, T> RandomAccessRange for Map<R, F, T> {}

/// An output expression that maps values in generated device code.
#[derive(Clone, Copy, Debug)]
pub struct MapOutput<R, F, Input, Output> {
    inner: R,
    map: F,
    values: PhantomData<fn(Input) -> Output>,
}

impl<R, F, Input, Output> MapOutput<R, F, Input, Output> {
    pub fn new(inner: R, map: F) -> Self {
        Self {
            inner,
            map,
            values: PhantomData,
        }
    }

    pub fn parts(&self) -> (&R, &F) {
        (&self.inner, &self.map)
    }
}

impl<R: DeviceRange, F, Input, Output> DeviceRange for MapOutput<R, F, Input, Output> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<R, F, Input, Output> WriteRange for MapOutput<R, F, Input, Output>
where
    R: WriteRange<Item = Output>,
    F: Fn(Input) -> Output,
    Input: DeviceCopy,
    Output: DeviceCopy,
{
    type Item = Input;
}

impl<R: RandomAccessRange, F, Input, Output> RandomAccessRange for MapOutput<R, F, Input, Output> {}

/// One expression with independent device read and write mappings.
#[derive(Clone, Copy, Debug)]
pub struct MapInOut<R, Read, Write, Input, Output> {
    inner: R,
    read: Read,
    write: Write,
    values: PhantomData<fn(Input) -> Output>,
}

impl<R, Read, Write, Input, Output> MapInOut<R, Read, Write, Input, Output> {
    pub fn new(inner: R, read: Read, write: Write) -> Self {
        Self {
            inner,
            read,
            write,
            values: PhantomData,
        }
    }

    pub fn parts(&self) -> (&R, &Read, &Write) {
        (&self.inner, &self.read, &self.write)
    }
}

impl<R: DeviceRange, Read, Write, Input, Output> DeviceRange
    for MapInOut<R, Read, Write, Input, Output>
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<R, Read, Write, Input, Output> ReadRange for MapInOut<R, Read, Write, Input, Output>
where
    R: ReadRange,
    Read: Fn(R::Item) -> Output,
    Output: DeviceCopy,
{
    type Item = Output;
}

impl<R, Read, Write, Input, Output> WriteRange for MapInOut<R, Read, Write, Input, Output>
where
    R: WriteRange,
    Write: Fn(Input) -> R::Item,
    Input: DeviceCopy,
{
    type Item = Input;
}

impl<R: RandomAccessRange, Read, Write, Input, Output> RandomAccessRange
    for MapInOut<R, Read, Write, Input, Output>
{
}

/// A product expression over two equal-length device ranges.
#[derive(Clone, Copy, Debug)]
pub struct Zip<A, B> {
    first: A,
    second: B,
    len: usize,
}

impl<A: DeviceRange, B: DeviceRange> Zip<A, B> {
    pub fn new(first: A, second: B) -> Option<Self> {
        let len = first.len();
        (second.len() == len).then_some(Self { first, second, len })
    }

    pub fn into_parts(self) -> (A, B) {
        (self.first, self.second)
    }

    fn shortest(first: A, second: B) -> Self {
        let len = first.len().min(second.len());
        Self { first, second, len }
    }
}

impl<A, B> DeviceRange for Zip<A, B> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<A: ReadRange, B: ReadRange> ReadRange for Zip<A, B>
where
    (A::Item, B::Item): DeviceCopy,
{
    type Item = (A::Item, B::Item);
}

impl<A: WriteRange, B: WriteRange> WriteRange for Zip<A, B>
where
    (A::Item, B::Item): DeviceCopy,
{
    type Item = (A::Item, B::Item);
}

impl<A: RandomAccessRange, B: RandomAccessRange> RandomAccessRange for Zip<A, B> {}

/// Indexed selection through a random-access `usize` range.
#[derive(Clone, Copy, Debug)]
pub struct Select<R, Indices> {
    inner: R,
    indices: Indices,
}

impl<R, Indices> Select<R, Indices> {
    pub fn new(inner: R, indices: Indices) -> Self {
        Self { inner, indices }
    }

    pub fn into_parts(self) -> (R, Indices) {
        (self.inner, self.indices)
    }
}

impl<R, Indices: DeviceRange> DeviceRange for Select<R, Indices> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<R, Indices> ReadRange for Select<R, Indices>
where
    R: ReadRange + RandomAccessRange,
    Indices: ReadRange<Item = usize> + RandomAccessRange,
{
    type Item = R::Item;
}

impl<R, Indices> RandomAccessRange for Select<R, Indices>
where
    R: RandomAccessRange,
    Indices: RandomAccessRange,
{
}

/// An index-aware output expression compiled into device code.
#[derive(Clone, Copy, Debug)]
pub struct IndexedOutput<F, T> {
    write: F,
    len: usize,
    value: PhantomData<fn(T)>,
}

impl<F, T> IndexedOutput<F, T> {
    pub fn new(len: usize, write: F) -> Self {
        Self {
            write,
            len,
            value: PhantomData,
        }
    }

    pub fn writer(&self) -> &F {
        &self.write
    }
}

impl<F, T> DeviceRange for IndexedOutput<F, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<F, T> WriteRange for IndexedOutput<F, T>
where
    F: Fn(usize, T),
    T: DeviceCopy,
{
    type Item = T;
}

impl<F, T> RandomAccessRange for IndexedOutput<F, T> {}

fn checked_bounds<R: RangeBounds<usize>>(
    bounds: R,
    len: usize,
) -> Result<(usize, usize), RangeError> {
    let start = match bounds.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.checked_add(1).ok_or(RangeError::BoundOverflow)?,
        Bound::Unbounded => 0,
    };
    let end = match bounds.end_bound() {
        Bound::Included(&value) => value.checked_add(1).ok_or(RangeError::BoundOverflow)?,
        Bound::Excluded(&value) => value,
        Bound::Unbounded => len,
    };
    if start > end || end > len {
        return Err(RangeError::OutOfBounds { start, end, len });
    }
    Ok((start, end))
}

/// Iterator-like construction of lazy finite device-range expressions.
///
/// These methods never evaluate a closure or access device storage on the
/// host. A device algorithm may accept the resulting generic type only after
/// that adaptor has a real generated lowering; construction alone does not
/// claim operational coverage.
pub trait DeviceRangeExt: DeviceRange + Sized {
    fn map<F, T>(self, map: F) -> Map<Self, F, T>
    where
        Self: ReadRange,
        F: Fn(Self::Item) -> T,
        T: DeviceCopy,
    {
        Map::new(self, map)
    }

    fn map_write<F, T>(self, map: F) -> MapOutput<Self, F, T, Self::Item>
    where
        Self: WriteRange,
        F: Fn(T) -> Self::Item,
        T: DeviceCopy,
    {
        MapOutput::new(self, map)
    }

    fn map_read_write<Read, Write, Input, Output>(
        self,
        read: Read,
        write: Write,
    ) -> MapInOut<Self, Read, Write, Input, Output>
    where
        Self: ReadRange + WriteRange,
        Read: Fn(<Self as ReadRange>::Item) -> Output,
        Write: Fn(Input) -> <Self as WriteRange>::Item,
        Input: DeviceCopy,
        Output: DeviceCopy,
    {
        MapInOut::new(self, read, write)
    }

    fn rev(self) -> Rev<Self> {
        Rev::new(self)
    }

    fn step_by(self, step: usize) -> Result<StepBy<Self>, RangeError> {
        StepBy::new(self, step).ok_or(RangeError::ZeroStep)
    }

    fn slice<R: RangeBounds<usize>>(self, bounds: R) -> Result<Subrange<Self>, RangeError> {
        let len = self.len();
        let (start, end) = checked_bounds(bounds, len)?;
        Subrange::new(self, start, end - start).ok_or(RangeError::OutOfBounds { start, end, len })
    }

    fn zip<R: DeviceRange>(self, other: R) -> Zip<Self, R> {
        Zip::shortest(self, other)
    }

    fn zip_eq<R: DeviceRange>(self, other: R) -> Result<Zip<Self, R>, RangeError> {
        let first = self.len();
        let second = other.len();
        Zip::new(self, other).ok_or(RangeError::LengthMismatch { first, second })
    }

    fn gather<I>(self, indices: I) -> Select<Self, I>
    where
        Self: ReadRange + RandomAccessRange,
        I: ReadRange<Item = usize> + RandomAccessRange,
    {
        Select::new(self, indices)
    }
}

impl<R: DeviceRange> DeviceRangeExt for R {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestInput<T> {
        len: usize,
        item: PhantomData<T>,
    }

    impl<T> TestInput<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                item: PhantomData,
            }
        }
    }

    impl<T> DeviceRange for TestInput<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T: DeviceCopy> ReadRange for TestInput<T> {
        type Item = T;
    }

    impl<T> RandomAccessRange for TestInput<T> {}
    impl<T> ContiguousRange for TestInput<T> {}

    struct TestOutput<T> {
        len: usize,
        item: PhantomData<T>,
    }

    impl<T> TestOutput<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                item: PhantomData,
            }
        }
    }

    impl<T> DeviceRange for TestOutput<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T: DeviceCopy> WriteRange for TestOutput<T> {
        type Item = T;
    }

    impl<T> RandomAccessRange for TestOutput<T> {}
    impl<T> ContiguousRange for TestOutput<T> {}

    struct TestInOut<T> {
        len: usize,
        item: PhantomData<T>,
    }

    impl<T> TestInOut<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                item: PhantomData,
            }
        }
    }

    impl<T> DeviceRange for TestInOut<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T: DeviceCopy> ReadRange for TestInOut<T> {
        type Item = T;
    }

    impl<T: DeviceCopy> WriteRange for TestInOut<T> {
        type Item = T;
    }

    impl<T> RandomAccessRange for TestInOut<T> {}

    fn read_i32<R: ReadRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn write_i32<R: WriteRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn read_pair<R: ReadRange<Item = (i32, i32)> + RandomAccessRange>(_range: R) {}

    #[test]
    fn range_expressions_expose_only_semantic_capabilities() {
        read_i32(repeat_n(7_i32, 3));
        read_i32(arithmetic_range(1_i32, 2_i32, 3));
        write_i32(discard::<i32>(3));

        let values = TestInput::<i32>::new(4);
        read_i32(values.slice(1..3).expect("valid subrange"));
        read_i32(values.rev());
        read_i32(values.step_by(2).expect("valid stride"));
        assert_eq!(values.step_by(0).err(), Some(RangeError::ZeroStep));
        assert!(matches!(
            values.slice(usize::MAX..),
            Err(RangeError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn construction_never_executes_user_closures_on_the_host() {
        let values = TestInput::<i32>::new(3);
        let mapped = values.map(|_value| std::panic::panic_any("map closure executed on host"));
        assert_eq!(mapped.len(), 3);
        read_i32(mapped);

        let tabulated = indexed_output::<_, i32>(3, |_index, _value| {
            std::panic::panic_any("indexed_output closure executed on host")
        });
        assert_eq!(tabulated.len(), 3);
        write_i32(tabulated);
    }

    #[test]
    fn composed_capabilities_preserve_shape_and_item_types() {
        let first = TestInput::<i32>::new(3);
        let second = TestInput::<i32>::new(3);
        let zipped = first.zip_eq(second).expect("equal lengths must zip");
        assert_eq!(zipped.len(), 3);
        read_pair(zipped);
        assert_eq!(first.zip(TestInput::<i32>::new(2)).len(), 2);
        assert!(matches!(
            first.zip_eq(TestInput::<i32>::new(2)),
            Err(RangeError::LengthMismatch {
                first: 3,
                second: 2
            })
        ));

        let indices = TestInput::<usize>::new(2);
        let selected = first.gather(indices);
        assert_eq!(selected.len(), 2);
        read_i32(selected);
    }

    #[test]
    fn writable_compositions_require_mutable_borrows() {
        let output = TestOutput::<i32>::new(4);
        write_i32(output.slice(1..3).expect("valid output subrange"));
        write_i32(TestOutput::<i32>::new(4).rev());
        write_i32(
            TestOutput::<i32>::new(4)
                .step_by(2)
                .expect("valid output stride"),
        );

        let mapped = TestOutput::new(4).map_write(|value: i32| value * 2);
        write_i32(mapped);

        let mapped = TestInOut::<i32>::new(4).map_read_write(|value| value + 1, |value| value - 1);
        read_i32(&mapped);
        write_i32(mapped);
    }

    #[test]
    fn included_and_unbounded_slice_bounds_match_rust_ranges() {
        let values = TestInput::<i32>::new(5);
        assert_eq!(values.slice(..=2).expect("valid prefix").len(), 3);
        assert_eq!(values.slice(2..).expect("valid suffix").len(), 3);
        assert_eq!(values.slice(..).expect("valid full range").len(), 5);
        assert!(matches!(
            values.slice(..=usize::MAX),
            Err(RangeError::BoundOverflow)
        ));
    }
}
