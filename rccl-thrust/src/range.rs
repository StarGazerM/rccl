use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use cuda_core::DeviceCopy;
use rccl_cub::{ContiguousRange, DeviceRange, RandomAccessRange, ReadRange, WriteRange};

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

/// A finite arithmetic device-range expression.
#[derive(Clone, Copy, Debug)]
pub struct ArithmeticRange<T> {
    start: T,
    step: T,
    len: usize,
}

impl<T> ArithmeticRange<T> {
    pub fn new(start: T, step: T, len: usize) -> Self {
        Self { start, step, len }
    }

    pub fn start(&self) -> &T {
        &self.start
    }

    pub fn step(&self) -> &T {
        &self.step
    }
}

impl<T> DeviceRange for ArithmeticRange<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> ReadRange for ArithmeticRange<T>
where
    T: DeviceCopy + TryFrom<usize> + Add<Output = T> + Mul<Output = T>,
{
    type Item = T;
}

impl<T> RandomAccessRange for ArithmeticRange<T> {}

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
/// closure; an algorithm macro compiles it into device code.
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
pub struct Tabulate<F, T> {
    write: F,
    len: usize,
    value: PhantomData<fn(T)>,
}

impl<F, T> Tabulate<F, T> {
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

impl<F, T> DeviceRange for Tabulate<F, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<F, T> WriteRange for Tabulate<F, T>
where
    F: FnMut(usize, T),
    T: DeviceCopy,
{
    type Item = T;
}

impl<F, T> RandomAccessRange for Tabulate<F, T> {}

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

    fn read_i32<R: ReadRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn write_i32<R: WriteRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn read_pair<R: ReadRange<Item = (i32, i32)> + RandomAccessRange>(_range: R) {}

    #[test]
    fn range_expressions_expose_only_semantic_capabilities() {
        read_i32(RepeatN::new(7_i32, 3));
        read_i32(ArithmeticRange::new(1_i32, 2_i32, 3));
        write_i32(Discard::<i32>::new(3));

        let values = TestInput::<i32>::new(4);
        read_i32(Subrange::new(values, 1, 2).expect("valid subrange"));
        read_i32(Rev::new(values));
        read_i32(StepBy::new(values, 2).expect("valid stride"));
        assert!(StepBy::new(values, 0).is_none());
        assert!(Subrange::new(values, usize::MAX, 2).is_none());
    }

    #[test]
    fn construction_never_executes_user_closures_on_the_host() {
        let values = TestInput::<i32>::new(3);
        let mapped = Map::<_, _, i32>::new(values, |_value| {
            std::panic::panic_any("map closure executed on host")
        });
        assert_eq!(mapped.len(), 3);
        read_i32(mapped);

        let tabulated = Tabulate::<_, i32>::new(3, |_index, _value| {
            std::panic::panic_any("tabulate closure executed on host")
        });
        assert_eq!(tabulated.len(), 3);
        write_i32(tabulated);
    }

    #[test]
    fn composed_capabilities_preserve_shape_and_item_types() {
        let first = TestInput::<i32>::new(3);
        let second = TestInput::<i32>::new(3);
        let zipped = Zip::new(first, second).expect("equal lengths must zip");
        assert_eq!(zipped.len(), 3);
        read_pair(zipped);
        assert!(Zip::new(first, TestInput::<i32>::new(2)).is_none());

        let indices = TestInput::<usize>::new(2);
        let selected = Select::new(first, indices);
        assert_eq!(selected.len(), 2);
        read_i32(selected);
    }

    #[test]
    fn writable_compositions_require_mutable_borrows() {
        let output = TestOutput::<i32>::new(4);
        write_i32(Subrange::new(output, 1, 2).expect("valid output subrange"));
        write_i32(Rev::new(TestOutput::<i32>::new(4)));
        write_i32(StepBy::new(TestOutput::<i32>::new(4), 2).expect("valid output stride"));

        let mapped = MapOutput::<_, _, i32, i32>::new(TestOutput::new(4), |value| value * 2);
        write_i32(mapped);
    }
}
