use crate::{ast::CubProgram, audit};
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::BTreeSet;
use syn::Result;

pub(crate) fn program(program: &CubProgram) -> Result<TokenStream> {
    require_support(
        program,
        [
            "double_buffer",
            "input_range",
            "key_value_pair",
            "output_range",
            "temp_storage",
        ],
    )?;

    audit::generated_rust(quote! {
        use core::mem::MaybeUninit;

        /// A key/value product with a named layout when tuple field names are
        /// semantically useful. Ordinary unnamed products use Rust tuples.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
        pub struct KeyValue<K, V> {
            pub key: K,
            pub value: V,
        }

        /// Two distinct mutable buffers used by ping-pong device algorithms.
        ///
        /// Construction is safe because Rust proves both exclusive borrows.
        pub struct DoubleBuffer<'a, T> {
            current: &'a mut [T],
            alternate: &'a mut [T],
            current_is_first: bool,
        }

        impl<'a, T> DoubleBuffer<'a, T> {
            pub fn new(current: &'a mut [T], alternate: &'a mut [T]) -> Option<Self> {
                (<[T]>::len(current) == <[T]>::len(alternate)).then_some(Self {
                    current,
                    alternate,
                    current_is_first: true,
                })
            }

            pub fn len(&self) -> usize {
                <[T]>::len(self.current)
            }

            pub fn is_empty(&self) -> bool {
                <[T]>::is_empty(self.current)
            }

            pub fn current(&self) -> &[T] {
                if self.current_is_first {
                    self.current
                } else {
                    self.alternate
                }
            }

            pub fn current_mut(&mut self) -> &mut [T] {
                if self.current_is_first {
                    self.current
                } else {
                    self.alternate
                }
            }

            pub fn alternate_mut(&mut self) -> &mut [T] {
                if self.current_is_first {
                    self.alternate
                } else {
                    self.current
                }
            }

            pub fn swap(&mut self) {
                self.current_is_first = !self.current_is_first;
            }
        }

        /// Caller-owned, aligned temporary storage expressed as uninitialized
        /// bytes. Algorithms interpret it only through generated safe views.
        pub struct TempStorage<'a> {
            bytes: &'a mut [MaybeUninit<u8>],
        }

        impl<'a> TempStorage<'a> {
            pub fn new(bytes: &'a mut [MaybeUninit<u8>]) -> Self {
                Self { bytes }
            }

            pub fn len(&self) -> usize {
                <[MaybeUninit<u8>]>::len(self.bytes)
            }

            pub fn is_empty(&self) -> bool {
                <[MaybeUninit<u8>]>::is_empty(self.bytes)
            }
        }

        /// Safe, bounds-checked indexed input used by generated device code.
        pub trait DeviceSource {
            type Item: Copy;

            fn len(&self) -> usize;
            fn get(&self, index: usize) -> Option<Self::Item>;

            fn is_empty(&self) -> bool {
                DeviceSource::len(self) == 0
            }
        }

        impl<T: Copy> DeviceSource for &[T] {
            type Item = T;

            fn len(&self) -> usize {
                <[T]>::len(self)
            }

            fn get(&self, index: usize) -> Option<T> {
                <[T]>::get(self, index).copied()
            }
        }

        impl<T: Copy> DeviceSource for &mut [T] {
            type Item = T;

            fn len(&self) -> usize {
                <[T]>::len(self)
            }

            fn get(&self, index: usize) -> Option<T> {
                <[T]>::get(self, index).copied()
            }
        }

        /// Safe, bounds-checked indexed output used by generated device code.
        ///
        /// Implementations must accept every `index < len()` and return `true`.
        /// Returning `false` propagates a violated sink contract to composed
        /// adaptors instead of silently dropping the error.
        pub trait DeviceSink<T> {
            fn len(&self) -> usize;

            fn is_empty(&self) -> bool {
                DeviceSink::len(self) == 0
            }

            fn write(&mut self, index: usize, value: T) -> bool;
        }

        impl<T> DeviceSink<T> for &mut [T] {
            fn len(&self) -> usize {
                <[T]>::len(self)
            }

            fn write(&mut self, index: usize, value: T) -> bool {
                let Some(slot) = <[T]>::get_mut(self, index) else {
                    return false;
                };
                *slot = value;
                true
            }
        }

        #[cfg(test)]
        mod generated_tests {
            use super::*;

            fn check(condition: bool) {
                if !condition {
                    std::panic::panic_any("generated CUB support contract failed");
                }
            }

            #[test]
            fn borrowed_buffers_preserve_aliasing_and_bounds() {
                let mut first = [1, 2, 3];
                let mut second = [4, 5, 6];
                {
                    let mut buffers = DoubleBuffer::new(&mut first, &mut second)
                        .expect("equal-length buffers must construct");
                    check(buffers.current() == [1, 2, 3]);
                    buffers.current_mut()[0] = 9;
                    buffers.swap();
                    check(buffers.current() == [4, 5, 6]);
                    buffers.alternate_mut()[2] = 8;
                }
                check(first == [9, 2, 8]);
                check(second == [4, 5, 6]);
                check(DoubleBuffer::new(&mut first[..1], &mut second[..2]).is_none());
            }

            #[test]
            fn sources_and_sinks_reject_out_of_bounds_without_side_effects() {
                let mut values = [10, 20, 30];
                {
                    let source = &mut values[..];
                    check(DeviceSource::get(&source, 2) == Some(30));
                    check(DeviceSource::get(&source, 3).is_none());
                }
                {
                    let mut sink = &mut values[..];
                    check(DeviceSink::write(&mut sink, 1, 7));
                    check(!DeviceSink::write(&mut sink, 3, 99));
                }
                check(values == [10, 7, 30]);
            }

            #[test]
            fn temporary_storage_is_only_a_borrowed_capacity() {
                let mut bytes = [MaybeUninit::<u8>::uninit(); 17];
                let storage = TempStorage::new(&mut bytes);
                check(storage.len() == 17);
                check(!storage.is_empty());
            }
        }
    })
}

fn require_support<'a>(
    program: &CubProgram,
    required: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let support = program
        .supports
        .iter()
        .map(|item| item.name.to_string())
        .collect::<BTreeSet<_>>();
    for name in required {
        if !support.contains(name) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("CUB DSL is missing required Rust mapping `{name}`"),
            ));
        }
    }
    Ok(())
}
