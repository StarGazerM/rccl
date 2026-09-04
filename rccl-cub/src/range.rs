use core::ops::{Add, Mul};
use cuda_core::DeviceCopy;
use rccl_memory::{DeviceAllocator, TemporaryValue};

/// A key/value product used when field names are semantically useful.
/// Ordinary products remain Rust tuples.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, DeviceCopy)]
pub struct KeyValue<K, V> {
    pub key: K,
    pub value: V,
}

/// Runtime extent shared by device-range expressions.
///
/// This trait contains no host-side access operation. A generated algorithm
/// backend lowers a concrete range expression to device access after Rust has
/// inferred its type.
pub trait DeviceRange {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        DeviceRange::len(self) == 0
    }
}

/// A range that generated device code may read.
pub trait ReadRange: DeviceRange {
    type Item: DeviceCopy;
}

/// A range that generated device code may write.
pub trait WriteRange: DeviceRange {
    type Item: DeviceCopy;
}

/// A range supporting constant-time indexed device access.
pub trait RandomAccessRange: DeviceRange {}

/// A random-access range with contiguous device representation.
pub trait ContiguousRange: RandomAccessRange {}

/// A finite arithmetic random-access device range.
///
/// This is the Rust-native replacement for a C++ counting iterator plus an
/// end iterator. Algorithms lower the value expression directly; it does not
/// allocate or materialize a device buffer.
#[derive(Clone, Copy, Debug)]
pub struct ArithmeticRange<T> {
    start: T,
    step: T,
    len: usize,
}

impl<T> ArithmeticRange<T> {
    pub const fn new(start: T, step: T, len: usize) -> Self {
        Self { start, step, len }
    }

    pub const fn start(&self) -> &T {
        &self.start
    }

    pub const fn step(&self) -> &T {
        &self.step
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub const fn arithmetic_range<T>(start: T, step: T, len: usize) -> ArithmeticRange<T> {
    ArithmeticRange::new(start, step, len)
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

impl<R: DeviceRange + ?Sized> DeviceRange for &R {
    fn len(&self) -> usize {
        DeviceRange::len(*self)
    }
}

impl<R: DeviceRange + ?Sized> DeviceRange for &mut R {
    fn len(&self) -> usize {
        DeviceRange::len(*self)
    }
}

impl<R: ReadRange + ?Sized> ReadRange for &R {
    type Item = R::Item;
}

impl<R: ReadRange + ?Sized> ReadRange for &mut R {
    type Item = R::Item;
}

impl<R: WriteRange + ?Sized> WriteRange for &mut R {
    type Item = R::Item;
}

impl<R: RandomAccessRange + ?Sized> RandomAccessRange for &R {}
impl<R: RandomAccessRange + ?Sized> RandomAccessRange for &mut R {}
impl<R: ContiguousRange + ?Sized> ContiguousRange for &R {}
impl<R: ContiguousRange + ?Sized> ContiguousRange for &mut R {}

impl<T: DeviceCopy> DeviceRange for cuda_core::DeviceBuffer<T> {
    fn len(&self) -> usize {
        cuda_core::DeviceBuffer::<T>::len(self)
    }
}

impl<T: DeviceCopy> ReadRange for cuda_core::DeviceBuffer<T> {
    type Item = T;
}

impl<T: DeviceCopy> WriteRange for cuda_core::DeviceBuffer<T> {
    type Item = T;
}

impl<T: DeviceCopy> RandomAccessRange for cuda_core::DeviceBuffer<T> {}
impl<T: DeviceCopy> ContiguousRange for cuda_core::DeviceBuffer<T> {}

impl<T, A> DeviceRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    fn len(&self) -> usize {
        rccl_memory::DeviceBuffer::<T, A>::len(self)
    }
}

impl<T, A> ReadRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Item = T;
}

impl<T, A> WriteRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Item = T;
}

impl<T, A> RandomAccessRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
}

impl<T, A> ContiguousRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
}

impl<T: TemporaryValue> DeviceRange for rccl_memory::TemporaryRegion<'_, T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: TemporaryValue> ReadRange for rccl_memory::TemporaryRegion<'_, T> {
    type Item = T;
}

impl<T: TemporaryValue> WriteRange for rccl_memory::TemporaryRegion<'_, T> {
    type Item = T;
}

impl<T: TemporaryValue> RandomAccessRange for rccl_memory::TemporaryRegion<'_, T> {}
impl<T: TemporaryValue> ContiguousRange for rccl_memory::TemporaryRegion<'_, T> {}

/// Device-side indexed read used by the generated range algorithm kernels.
///
/// This is a lowering interface, not a host iterator. Implementations are
/// passed to CUDA by value and their methods are called only from device code.
#[doc(hidden)]
pub trait DeviceReadAt: Copy {
    type Item: Copy;

    /// Read the item at a previously bounds-checked logical index.
    ///
    /// # Safety
    ///
    /// `index` must be in the logical range and every referenced allocation
    /// must remain alive and immutable for the kernel operation.
    unsafe fn read_at(self, index: usize) -> Self::Item;
}

/// Device-side indexed write used by the generated range algorithm kernels.
///
/// Safe host APIs only construct these values while retaining the affine
/// writable range from which they were lowered.
#[doc(hidden)]
pub trait DeviceWriteAt: Copy {
    type Item: Copy;

    /// Write an item at a previously bounds-checked logical index.
    ///
    /// # Safety
    ///
    /// `index` must be in the logical range, the destination must be valid,
    /// and concurrently executing indices must not alias.
    unsafe fn write_at(self, index: usize, value: Self::Item);
}

/// Dense device output whose base pointer is available to vectorized kernels.
#[doc(hidden)]
pub trait DenseDeviceWriteAt: DeviceWriteAt {
    fn as_mut_ptr(self) -> *mut Self::Item;
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceSliceRead<T> {
    pointer: *const T,
}

impl<T> DeviceSliceRead<T> {
    pub const fn new(pointer: *const T) -> Self {
        Self { pointer }
    }
}

impl<T: Copy> DeviceReadAt for DeviceSliceRead<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> T {
        // SAFETY: guaranteed by DeviceReadAt::read_at's caller contract.
        unsafe { self.pointer.add(index).read() }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceSliceWrite<T> {
    pointer: *mut T,
}

impl<T> DeviceSliceWrite<T> {
    pub const fn new(pointer: *mut T) -> Self {
        Self { pointer }
    }
}

impl<T: Copy> DeviceReadAt for DeviceSliceWrite<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> T {
        // SAFETY: guaranteed by DeviceReadAt::read_at's caller contract.
        unsafe { self.pointer.add(index).read() }
    }
}

impl<T: Copy> DeviceWriteAt for DeviceSliceWrite<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: T) {
        // SAFETY: guaranteed by DeviceWriteAt::write_at's caller contract.
        unsafe { self.pointer.add(index).write(value) };
    }
}

impl<T: Copy> DenseDeviceWriteAt for DeviceSliceWrite<T> {
    #[inline(always)]
    fn as_mut_ptr(self) -> *mut T {
        self.pointer
    }
}

/// Primitive arithmetic used by device counting ranges without debug-mode
/// overflow branches in generated PTX.
#[doc(hidden)]
pub trait DeviceArithmetic: Copy {
    fn index_fits(index: usize) -> bool;

    fn value_at(start: Self, step: Self, index: usize) -> Self;
}

macro_rules! impl_device_arithmetic {
    ($($ty:ty),* $(,)?) => {
        $(
            impl DeviceArithmetic for $ty {
                #[inline(always)]
                fn index_fits(index: usize) -> bool {
                    index <= <$ty>::MAX as usize
                }

                #[inline(always)]
                fn value_at(start: Self, step: Self, index: usize) -> Self {
                    start.wrapping_add(step.wrapping_mul(index as Self))
                }
            }
        )*
    };
}

impl_device_arithmetic!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Device descriptor for an arithmetic range. The host-only extent is omitted
/// because the kernel entry carries its validated launch extent separately.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceArithmeticRange<T> {
    start: T,
    step: T,
}

impl<T> DeviceArithmeticRange<T> {
    pub const fn new(start: T, step: T) -> Self {
        Self { start, step }
    }
}

impl<T: DeviceArithmetic> DeviceReadAt for DeviceArithmeticRange<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> T {
        T::value_at(self.start, self.step, index)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceConstant<T> {
    value: T,
}

impl<T> DeviceConstant<T> {
    pub const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Copy> DeviceReadAt for DeviceConstant<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, _index: usize) -> T {
        self.value
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceMap<R, F> {
    inner: R,
    map: F,
}

impl<R, F> DeviceMap<R, F> {
    pub const fn new(inner: R, map: F) -> Self {
        Self { inner, map }
    }
}

impl<R, F, T> DeviceReadAt for DeviceMap<R, F>
where
    R: DeviceReadAt,
    F: Fn(R::Item) -> T + Copy,
    T: Copy,
{
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> T {
        // SAFETY: this adaptor preserves the validated logical index.
        (self.map)(unsafe { self.inner.read_at(index) })
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceMapWrite<R, F, Input> {
    inner: R,
    map: F,
    input: core::marker::PhantomData<fn(Input)>,
}

impl<R, F, Input> DeviceMapWrite<R, F, Input> {
    pub const fn new(inner: R, map: F) -> Self {
        Self {
            inner,
            map,
            input: core::marker::PhantomData,
        }
    }
}

impl<R, F, Input> DeviceWriteAt for DeviceMapWrite<R, F, Input>
where
    R: DeviceWriteAt,
    F: Fn(Input) -> R::Item + Copy,
    Input: Copy,
{
    type Item = Input;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Input) {
        // SAFETY: this adaptor preserves the validated logical index and
        // affine destination.
        unsafe { self.inner.write_at(index, (self.map)(value)) };
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceMapInOut<R, Read, Write, Input, Output> {
    inner: R,
    read: Read,
    write: Write,
    values: core::marker::PhantomData<fn(Input) -> Output>,
}

impl<R, Read, Write, Input, Output> DeviceMapInOut<R, Read, Write, Input, Output> {
    pub const fn new(inner: R, read: Read, write: Write) -> Self {
        Self {
            inner,
            read,
            write,
            values: core::marker::PhantomData,
        }
    }
}

impl<R, Read, Write, Input, Output> DeviceReadAt for DeviceMapInOut<R, Read, Write, Input, Output>
where
    R: DeviceReadAt,
    Read: Fn(R::Item) -> Output + Copy,
    Write: Copy,
    Input: Copy,
    Output: Copy,
{
    type Item = Output;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Output {
        // SAFETY: this adaptor preserves the validated logical index.
        (self.read)(unsafe { self.inner.read_at(index) })
    }
}

impl<R, Read, Write, Input, Output> DeviceWriteAt for DeviceMapInOut<R, Read, Write, Input, Output>
where
    R: DeviceWriteAt,
    Read: Copy,
    Write: Fn(Input) -> R::Item + Copy,
    Input: Copy,
    Output: Copy,
{
    type Item = Input;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Input) {
        // SAFETY: this adaptor preserves the validated logical index and
        // affine destination.
        unsafe { self.inner.write_at(index, (self.write)(value)) };
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceZip<A, B> {
    first: A,
    second: B,
}

impl<A, B> DeviceZip<A, B> {
    pub const fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> DeviceReadAt for DeviceZip<A, B>
where
    A: DeviceReadAt,
    B: DeviceReadAt,
{
    type Item = (A::Item, B::Item);

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Self::Item {
        // SAFETY: zip construction validates both logical extents.
        unsafe { (self.first.read_at(index), self.second.read_at(index)) }
    }
}

impl<A, B> DeviceWriteAt for DeviceZip<A, B>
where
    A: DeviceWriteAt,
    B: DeviceWriteAt,
{
    type Item = (A::Item, B::Item);

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Self::Item) {
        // SAFETY: zip preserves each affine destination and logical index.
        unsafe {
            self.first.write_at(index, value.0);
            self.second.write_at(index, value.1);
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceOffset<R> {
    inner: R,
    offset: usize,
}

impl<R> DeviceOffset<R> {
    pub const fn new(inner: R, offset: usize) -> Self {
        Self { inner, offset }
    }
}

impl<R: DeviceReadAt> DeviceReadAt for DeviceOffset<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Self::Item {
        // SAFETY: checked subrange construction proves this addition and bound.
        unsafe { self.inner.read_at(self.offset + index) }
    }
}

impl<R: DeviceWriteAt> DeviceWriteAt for DeviceOffset<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Self::Item) {
        // SAFETY: checked subrange construction proves this addition and bound.
        unsafe { self.inner.write_at(self.offset + index, value) };
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceReverse<R> {
    inner: R,
    len: usize,
}

impl<R> DeviceReverse<R> {
    pub const fn new(inner: R, len: usize) -> Self {
        Self { inner, len }
    }
}

impl<R: DeviceReadAt> DeviceReadAt for DeviceReverse<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Self::Item {
        // SAFETY: a launched index is smaller than len, so subtraction cannot
        // underflow and the translated index remains in range.
        unsafe { self.inner.read_at(self.len - 1 - index) }
    }
}

impl<R: DeviceWriteAt> DeviceWriteAt for DeviceReverse<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Self::Item) {
        // SAFETY: see DeviceReverse::read_at.
        unsafe { self.inner.write_at(self.len - 1 - index, value) };
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceStride<R> {
    inner: R,
    step: usize,
}

impl<R> DeviceStride<R> {
    pub const fn new(inner: R, step: usize) -> Self {
        Self { inner, step }
    }
}

impl<R: DeviceReadAt> DeviceReadAt for DeviceStride<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Self::Item {
        // SAFETY: host validation proves the multiplication and translated
        // range for every launched index.
        unsafe { self.inner.read_at(index * self.step) }
    }
}

impl<R: DeviceWriteAt> DeviceWriteAt for DeviceStride<R> {
    type Item = R::Item;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: Self::Item) {
        // SAFETY: host validation proves this affine destination index.
        unsafe { self.inner.write_at(index * self.step, value) };
    }
}

/// Integer accepted by a device permutation descriptor.
///
/// RCCL workloads use `u32` row identifiers for the common path and `u64` or
/// `usize` when the relation extent requires it. Signed C++ iterator index
/// types are deliberately not reproduced.
#[doc(hidden)]
pub trait DeviceIndex: Copy {
    fn to_usize(self) -> usize;
}

impl DeviceIndex for u32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl DeviceIndex for u64 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        if self > usize::MAX as u64 {
            cuda_device::debug::trap();
        }
        self as usize
    }
}

impl DeviceIndex for usize {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DevicePermutation<R, Indices> {
    inner: R,
    indices: Indices,
    source_len: usize,
}

impl<R, Indices> DevicePermutation<R, Indices> {
    pub const fn new(inner: R, indices: Indices, source_len: usize) -> Self {
        Self {
            inner,
            indices,
            source_len,
        }
    }
}

impl<R, Indices> DeviceReadAt for DevicePermutation<R, Indices>
where
    R: DeviceReadAt,
    Indices: DeviceReadAt,
    Indices::Item: DeviceIndex,
{
    type Item = R::Item;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Self::Item {
        // SAFETY: the transform kernel bounds `index` by the index-range
        // extent. Every selected source position is checked here before the
        // underlying descriptor is accessed.
        let selected = unsafe { self.indices.read_at(index) }.to_usize();
        if selected >= self.source_len {
            cuda_device::debug::trap();
        }
        unsafe { self.inner.read_at(selected) }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceDiscard<T>(core::marker::PhantomData<fn(T)>);

impl<T> DeviceDiscard<T> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<T: Copy> DeviceWriteAt for DeviceDiscard<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn write_at(self, _index: usize, _value: T) {}
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DeviceIndexedWrite<F, T> {
    write: F,
    item: core::marker::PhantomData<fn(T)>,
}

impl<F, T> DeviceIndexedWrite<F, T> {
    pub const fn new(write: F) -> Self {
        Self {
            write,
            item: core::marker::PhantomData,
        }
    }
}

impl<F, T> DeviceWriteAt for DeviceIndexedWrite<F, T>
where
    F: Fn(usize, T) + Copy,
    T: Copy,
{
    type Item = T;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: T) {
        (self.write)(index, value);
    }
}
