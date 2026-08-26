use crate::{ast::ThrustProgram, audit};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Result;

pub(crate) fn program(program: &ThrustProgram) -> Result<TokenStream> {
    for required in ["execution_policy", "system_error"] {
        if !program
            .supports
            .iter()
            .any(|support| support.name == required)
        {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("Thrust DSL is missing required Rust mapping `{required}`"),
            ));
        }
    }
    require_implemented_adaptors(
        program,
        [
            "constant_iterator",
            "counting_iterator",
            "discard_iterator",
            "offset_iterator",
            "permutation_iterator",
            "reverse_iterator",
            "shuffle_iterator",
            "strided_iterator",
            "tabulate_output_iterator",
            "transform_input_output_iterator",
            "transform_iterator",
            "transform_output_iterator",
            "zip_iterator",
        ],
    )?;

    audit::generated_rust(quote! {
        pub use cuda_core::DeviceCopy;
        pub use rccl_cub::{DeviceSink, DeviceSource, DoubleBuffer, KeyValue, TempStorage};
        pub use rccl_memory as memory;

        pub mod range {
            use core::{
                marker::PhantomData,
                ops::{Add, Mul},
            };
            use rccl_cub::{DeviceSink, DeviceSource};

            /// A finite source that yields one copied value at every index.
            #[derive(Clone, Copy, Debug)]
            pub struct RepeatN<T> {
                value: T,
                len: usize,
            }

            impl<T> RepeatN<T> {
                pub fn new(value: T, len: usize) -> Self {
                    Self { value, len }
                }
            }

            impl<T: Copy> DeviceSource for RepeatN<T> {
                type Item = T;

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<T> {
                    (index < self.len).then_some(self.value)
                }
            }

            /// A finite arithmetic progression `start + step * index`.
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
            }

            impl<T> DeviceSource for ArithmeticRange<T>
            where
                T: Copy + TryFrom<usize> + Add<Output = T> + Mul<Output = T>,
            {
                type Item = T;

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<T> {
                    if index >= self.len {
                        return None;
                    }
                    let index = T::try_from(index).ok()?;
                    Some(self.start + self.step * index)
                }
            }

            /// A sink that validates the index and intentionally drops values.
            #[derive(Clone, Copy, Debug)]
            pub struct Discard {
                len: usize,
            }

            impl Discard {
                pub fn new(len: usize) -> Self {
                    Self { len }
                }
            }

            impl<T> DeviceSink<T> for Discard {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, _value: T) -> bool {
                    index < self.len
                }
            }

            /// A checked contiguous logical subrange of another source/sink.
            #[derive(Clone, Copy, Debug)]
            pub struct Subrange<R> {
                inner: R,
                start: usize,
                len: usize,
            }

            impl<R> Subrange<R> {
                pub fn into_inner(self) -> R {
                    self.inner
                }
            }

            impl<R: DeviceSource> Subrange<R> {
                pub fn source(inner: R, start: usize, len: usize) -> Option<Self> {
                    let end = start.checked_add(len)?;
                    (end <= inner.len()).then_some(Self { inner, start, len })
                }
            }

            impl<R> Subrange<R> {
                pub fn sink<T>(inner: R, start: usize, len: usize) -> Option<Self>
                where
                    R: DeviceSink<T>,
                {
                    let end = start.checked_add(len)?;
                    (end <= DeviceSink::<T>::len(&inner)).then_some(Self { inner, start, len })
                }
            }

            impl<R: DeviceSource> DeviceSource for Subrange<R> {
                type Item = R::Item;

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<Self::Item> {
                    (index < self.len)
                        .then(|| self.start + index)
                        .and_then(|index| self.inner.get(index))
                }
            }

            impl<R: DeviceSink<T>, T> DeviceSink<T> for Subrange<R> {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, value: T) -> bool {
                    index < self.len && self.inner.write(self.start + index, value)
                }
            }

            /// A reversed logical view.
            #[derive(Clone, Copy, Debug)]
            pub struct Rev<R> {
                inner: R,
                len: usize,
            }

            impl<R: DeviceSource> Rev<R> {
                pub fn source(inner: R) -> Self {
                    let len = inner.len();
                    Self { inner, len }
                }
            }

            impl<R> Rev<R> {
                pub fn sink<T>(inner: R) -> Self
                where
                    R: DeviceSink<T>,
                {
                    let len = DeviceSink::<T>::len(&inner);
                    Self { inner, len }
                }
            }

            impl<R: DeviceSource> DeviceSource for Rev<R> {
                type Item = R::Item;

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<Self::Item> {
                    (index < self.len)
                        .then(|| self.len - 1 - index)
                        .and_then(|index| self.inner.get(index))
                }
            }

            impl<R: DeviceSink<T>, T> DeviceSink<T> for Rev<R> {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, value: T) -> bool {
                    index < self.len && self.inner.write(self.len - 1 - index, value)
                }
            }

            /// A view that keeps every `step`th element.
            #[derive(Clone, Copy, Debug)]
            pub struct StepBy<R> {
                inner: R,
                step: usize,
                len: usize,
            }

            impl<R> StepBy<R> {
                fn shape(inner_len: usize, step: usize) -> Option<(usize, usize)> {
                    if step == 0 {
                        return None;
                    }
                    Some((step, inner_len.div_ceil(step)))
                }
            }

            impl<R: DeviceSource> StepBy<R> {
                pub fn source(inner: R, step: usize) -> Option<Self> {
                    let (step, len) = Self::shape(inner.len(), step)?;
                    Some(Self { inner, step, len })
                }
            }

            impl<R> StepBy<R> {
                pub fn sink<T>(inner: R, step: usize) -> Option<Self>
                where
                    R: DeviceSink<T>,
                {
                    let (step, len) = Self::shape(DeviceSink::<T>::len(&inner), step)?;
                    Some(Self { inner, step, len })
                }
            }

            impl<R: DeviceSource> DeviceSource for StepBy<R> {
                type Item = R::Item;

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<Self::Item> {
                    (index < self.len)
                        .then(|| index * self.step)
                        .and_then(|index| self.inner.get(index))
                }
            }

            impl<R: DeviceSink<T>, T> DeviceSink<T> for StepBy<R> {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, value: T) -> bool {
                    index < self.len && self.inner.write(index * self.step, value)
                }
            }

            /// A zero-copy mapped source.
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
            }

            impl<R, F, T> DeviceSource for Map<R, F, T>
            where
                R: DeviceSource,
                F: Fn(R::Item) -> T,
                T: Copy,
            {
                type Item = T;

                fn len(&self) -> usize {
                    self.inner.len()
                }

                fn get(&self, index: usize) -> Option<T> {
                    self.inner.get(index).map(&self.map)
                }
            }

            /// A sink that maps a value before forwarding it.
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
            }

            impl<R, F, Input, Output> DeviceSink<Input> for MapOutput<R, F, Input, Output>
            where
                R: DeviceSink<Output>,
                F: Fn(Input) -> Output,
            {
                fn len(&self) -> usize {
                    self.inner.len()
                }

                fn write(&mut self, index: usize, value: Input) -> bool {
                    self.inner.write(index, (self.map)(value))
                }
            }

            /// One range with independent read and write mappings.
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
            }

            impl<R, Read, Write, Input, Output> DeviceSource
                for MapInOut<R, Read, Write, Input, Output>
            where
                R: DeviceSource,
                Read: Fn(R::Item) -> Output,
                Output: Copy,
            {
                type Item = Output;

                fn len(&self) -> usize {
                    self.inner.len()
                }

                fn get(&self, index: usize) -> Option<Output> {
                    self.inner.get(index).map(&self.read)
                }
            }

            impl<R, Read, Write, Input, Output> DeviceSink<Input>
                for MapInOut<R, Read, Write, Input, Output>
            where
                R: DeviceSource + DeviceSink<<R as DeviceSource>::Item>,
                Write: Fn(Input) -> <R as DeviceSource>::Item,
            {
                fn len(&self) -> usize {
                    DeviceSink::len(&self.inner)
                }

                fn write(&mut self, index: usize, value: Input) -> bool {
                    self.inner.write(index, (self.write)(value))
                }
            }

            /// A product of two equal-length sources or sinks.
            #[derive(Clone, Copy, Debug)]
            pub struct Zip<A, B> {
                first: A,
                second: B,
                len: usize,
            }

            impl<A: DeviceSource, B: DeviceSource> Zip<A, B> {
                pub fn sources(first: A, second: B) -> Option<Self> {
                    let len = first.len();
                    (second.len() == len).then_some(Self { first, second, len })
                }
            }

            impl<A, B> Zip<A, B> {
                pub fn sinks<T, U>(first: A, second: B) -> Option<Self>
                where
                    A: DeviceSink<T>,
                    B: DeviceSink<U>,
                {
                    let len = first.len();
                    (second.len() == len).then_some(Self { first, second, len })
                }
            }

            impl<A: DeviceSource, B: DeviceSource> DeviceSource for Zip<A, B> {
                type Item = (A::Item, B::Item);

                fn len(&self) -> usize {
                    self.len
                }

                fn get(&self, index: usize) -> Option<Self::Item> {
                    Some((self.first.get(index)?, self.second.get(index)?))
                }
            }

            impl<A: DeviceSink<T>, B: DeviceSink<U>, T, U> DeviceSink<(T, U)> for Zip<A, B> {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, value: (T, U)) -> bool {
                    if index >= self.len {
                        return false;
                    }
                    let (first, second) = value;
                    let first_written = self.first.write(index, first);
                    let second_written = self.second.write(index, second);
                    first_written && second_written
                }
            }

            /// Indexed selection through a source of `usize` positions.
            #[derive(Clone, Copy, Debug)]
            pub struct Select<R, Indices> {
                inner: R,
                indices: Indices,
            }

            impl<R, Indices> Select<R, Indices> {
                pub fn new(inner: R, indices: Indices) -> Self {
                    Self { inner, indices }
                }
            }

            impl<R, Indices> DeviceSource for Select<R, Indices>
            where
                R: DeviceSource,
                Indices: DeviceSource<Item = usize>,
            {
                type Item = R::Item;

                fn len(&self) -> usize {
                    self.indices.len()
                }

                fn get(&self, index: usize) -> Option<Self::Item> {
                    self.inner.get(self.indices.get(index)?)
                }
            }

            /// An output defined by an index-aware closure.
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
            }

            impl<F, T> DeviceSink<T> for Tabulate<F, T>
            where
                F: FnMut(usize, T),
            {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, index: usize, value: T) -> bool {
                    if index >= self.len {
                        return false;
                    }
                    (self.write)(index, value);
                    true
                }
            }
        }

        #[cfg(test)]
        mod generated_tests {
            use super::range::*;
            use rccl_cub::{DeviceSink, DeviceSource};

            fn check(condition: bool) {
                if !condition {
                    std::panic::panic_any("generated Thrust range contract failed");
                }
            }

            struct RejectSink {
                len: usize,
            }

            impl<T> DeviceSink<T> for RejectSink {
                fn len(&self) -> usize {
                    self.len
                }

                fn write(&mut self, _index: usize, _value: T) -> bool {
                    false
                }
            }

            #[test]
            fn finite_sources_are_exact_and_checked() {
                let repeated = RepeatN::new(7_i32, 3);
                check(repeated.len() == 3);
                check(repeated.get(2) == Some(7));
                check(repeated.get(3).is_none());

                let counted = ArithmeticRange::new(5_i32, 3_i32, 4);
                check(counted.get(0) == Some(5));
                check(counted.get(3) == Some(14));
                check(counted.get(4).is_none());
            }

            #[test]
            fn subrange_reverse_and_stride_preserve_boundaries() {
                let values = [0_i32, 1, 2, 3, 4];
                let subrange = Subrange::source(&values[..], 1, 3)
                    .expect("an in-bounds subrange must construct");
                check(subrange.get(0) == Some(1));
                check(subrange.get(2) == Some(3));
                check(subrange.get(3).is_none());
                check(Subrange::source(&values[..], usize::MAX, 2).is_none());
                check(Subrange::source(&values[..], 4, 2).is_none());

                let reversed = Rev::source(&values[..]);
                check(reversed.get(0) == Some(4));
                check(reversed.get(4) == Some(0));
                check(reversed.get(5).is_none());

                let stepped = StepBy::source(&values[..], 2)
                    .expect("a nonzero stride must construct");
                check(stepped.len() == 3);
                check(stepped.get(0) == Some(0));
                check(stepped.get(2) == Some(4));
                check(stepped.get(3).is_none());
                check(StepBy::source(&values[..], 0).is_none());
            }

            #[test]
            fn closure_adaptors_use_rust_closures_and_propagate_failure() {
                let bias = 11_i32;
                let mapped = Map::<_, _, i32>::new(RepeatN::new(4_i32, 2), |value| value + bias);
                check(mapped.get(1) == Some(15));

                let mut output = [0_i32; 2];
                {
                    let mut mapped_output = MapOutput::<_, _, i32, i32>::new(
                        &mut output[..],
                        |value| value * 2,
                    );
                    check(mapped_output.write(1, 6));
                    check(!mapped_output.write(2, 100));
                }
                check(output == [0, 12]);

                let mut in_out = [1_i32, 2, 3];
                {
                    let mut mapped = MapInOut::<_, _, _, i32, i32>::new(
                        &mut in_out[..],
                        |value| value + 10,
                        |value| value * 2,
                    );
                    check(DeviceSource::get(&mapped, 1) == Some(12));
                    check(DeviceSink::write(&mut mapped, 2, 7));
                    check(!DeviceSink::write(&mut mapped, 3, 9));
                }
                check(in_out == [1, 2, 14]);
            }

            #[test]
            fn writable_views_preserve_positions_and_report_component_rejection() {
                let mut output = [0_i32; 6];
                {
                    let mut subrange = Subrange::sink::<i32>(&mut output[..], 2, 3)
                        .expect("an in-bounds output subrange must construct");
                    check(subrange.write(0, 4));
                    check(subrange.write(2, 6));
                    check(!subrange.write(3, 9));
                }
                check(output == [0, 0, 4, 0, 6, 0]);

                {
                    let mut reversed = Rev::sink::<i32>(&mut output[..]);
                    check(reversed.write(0, 8));
                    check(reversed.write(5, 7));
                    check(!reversed.write(6, 5));
                }
                check(output == [7, 0, 4, 0, 6, 8]);

                {
                    let mut stepped = StepBy::sink::<i32>(&mut output[..], 2)
                        .expect("a nonzero output stride must construct");
                    check(stepped.write(1, 3));
                    check(!stepped.write(3, 9));
                }
                check(output == [7, 0, 3, 0, 6, 8]);

                let mut second = [0_i32; 1];
                {
                    let mut zipped = Zip::sinks::<i32, i32>(
                        RejectSink { len: 1 },
                        &mut second[..],
                    )
                    .expect("equal sink lengths must zip");
                    check(!zipped.write(0, (1, 2)));
                }
                check(second == [2]);
            }

            #[test]
            fn product_permutation_and_tabulation_do_not_mask_errors() {
                let first = [1_i32, 2, 3];
                let second = [4_i32, 5, 6];
                let zipped = Zip::sources(&first[..], &second[..])
                    .expect("equal source lengths must zip");
                check(zipped.get(1) == Some((2, 5)));
                check(Zip::sources(&first[..], &second[..2]).is_none());

                let indices = [2_usize, 0, 9];
                let selected = Select::new(&first[..], &indices[..]);
                check(selected.get(0) == Some(3));
                check(selected.get(1) == Some(1));
                check(selected.get(2).is_none());

                let mut writes = 0_usize;
                {
                    let mut tabulated = Tabulate::new(1, |_index, _value: i32| writes += 1);
                    check(!tabulated.write(1, 8));
                    check(tabulated.write(0, 8));
                }
                check(writes == 1);

                let mut discarded = Discard::new(1);
                check(discarded.write(0, 1_i32));
                check(!discarded.write(1, 1_i32));
            }
        }

        pub mod cuda {
            use cuda_core::CudaStream;
            use rccl_memory::{DeviceAllocator, MemoryError};
            use std::{error::Error, fmt};

            /// A borrowed CUDA stream and borrowed allocator.
            ///
            /// It contains no algorithm, type, target, schedule, module set,
            /// compiler manifest, or cache state.
            #[derive(Clone, Copy)]
            pub struct Execution<'stream, 'allocator, A: DeviceAllocator + ?Sized> {
                stream: &'stream CudaStream,
                allocator: &'allocator A,
            }

            impl<A: DeviceAllocator + ?Sized> Execution<'_, '_, A> {
                pub fn stream(&self) -> &CudaStream {
                    self.stream
                }

                pub fn allocator(&self) -> &A {
                    self.allocator
                }
            }

            #[derive(Debug)]
            pub struct ExecutionError(MemoryError);

            impl fmt::Display for ExecutionError {
                fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.0.fmt(formatter)
                }
            }

            impl Error for ExecutionError {
                fn source(&self) -> Option<&(dyn Error + 'static)> {
                    Some(&self.0)
                }
            }

            /// Borrow a stream and allocator for generated Thrust calls.
            pub fn par<'stream, 'allocator, A: DeviceAllocator + ?Sized>(
                stream: &'stream CudaStream,
                allocator: &'allocator A,
            ) -> Result<Execution<'stream, 'allocator, A>, ExecutionError> {
                if allocator.context().as_ref() != stream.context().as_ref() {
                    return Err(ExecutionError(MemoryError::ContextMismatch));
                }
                Ok(Execution { stream, allocator })
            }
        }
    })
}

fn require_implemented_adaptors<'a>(
    program: &ThrustProgram,
    required: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    for name in required {
        let Some(adaptor) = program.adaptors.iter().find(|adaptor| adaptor.name == name) else {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("Thrust DSL is missing generated adaptor `{name}`"),
            ));
        };
        if adaptor.status != "implemented" {
            return Err(syn::Error::new(
                adaptor.status.span(),
                format!("generated adaptor `{name}` must have status `implemented`"),
            ));
        }
    }
    Ok(())
}
