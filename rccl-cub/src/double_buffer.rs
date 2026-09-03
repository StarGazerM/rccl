use crate::DeviceRange;

/// Two same-shaped ranges used by ping-pong algorithms.
///
/// Passing two mutable borrows makes exclusivity a normal Rust property.
pub struct DoubleBuffer<First, Second = First> {
    first: First,
    second: Second,
    current_is_first: bool,
    len: usize,
}

impl<First: DeviceRange, Second: DeviceRange> DoubleBuffer<First, Second> {
    pub fn new(first: First, second: Second) -> Option<Self> {
        let len = first.len();
        (second.len() == len).then_some(Self {
            first,
            second,
            current_is_first: true,
            len,
        })
    }
}

impl<FirstKey, FirstValue, SecondKey, SecondValue>
    DoubleBuffer<(FirstKey, FirstValue), (SecondKey, SecondValue)>
where
    FirstKey: DeviceRange,
    FirstValue: DeviceRange,
    SecondKey: DeviceRange,
    SecondValue: DeviceRange,
{
    /// Construct one selector over paired key/value ranges.
    ///
    /// A single selector makes key/value synchronization structural rather
    /// than a convention between independently mutable selector integers.
    pub fn paired(first: (FirstKey, FirstValue), second: (SecondKey, SecondValue)) -> Option<Self> {
        let len = first.0.len();
        (first.1.len() == len && second.0.len() == len && second.1.len() == len).then_some(Self {
            first,
            second,
            current_is_first: true,
            len,
        })
    }
}

impl<First, Second> DoubleBuffer<First, Second> {
    #[doc(hidden)]
    pub fn __from_equal_parts(first: First, second: Second, len: usize) -> Self {
        Self {
            first,
            second,
            current_is_first: true,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn current_is_first(&self) -> bool {
        self.current_is_first
    }

    pub fn first(&self) -> &First {
        &self.first
    }

    pub fn second(&self) -> &Second {
        &self.second
    }

    pub fn first_and_second_mut(&mut self) -> (&mut First, &mut Second) {
        (&mut self.first, &mut self.second)
    }

    pub fn swap(&mut self) {
        self.current_is_first = !self.current_is_first;
    }

    pub fn into_parts(self) -> (First, Second) {
        (self.first, self.second)
    }
}

impl<R> DoubleBuffer<R, R> {
    pub fn current(&self) -> &R {
        if self.current_is_first {
            &self.first
        } else {
            &self.second
        }
    }

    pub fn current_mut(&mut self) -> &mut R {
        if self.current_is_first {
            &mut self.first
        } else {
            &mut self.second
        }
    }

    pub fn alternate_mut(&mut self) -> &mut R {
        if self.current_is_first {
            &mut self.second
        } else {
            &mut self.first
        }
    }

    /// Borrow the selected input and alternate output together.
    ///
    /// The two references are disjoint because they come from different
    /// fields. This is the Rust replacement for CUB's pointer/selector pair:
    /// generated algorithms never recover either address or manufacture an
    /// alias.
    pub fn current_and_alternate_mut(&mut self) -> (&mut R, &mut R) {
        if self.current_is_first {
            (&mut self.first, &mut self.second)
        } else {
            (&mut self.second, &mut self.first)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContiguousRange, RandomAccessRange, ReadRange, WriteRange};
    use core::marker::PhantomData;
    use cuda_core::DeviceCopy;

    struct TestRange<T> {
        len: usize,
        tag: usize,
        item: PhantomData<T>,
    }

    impl<T> TestRange<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                tag: 0,
                item: PhantomData,
            }
        }
    }

    impl<T> DeviceRange for TestRange<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T: DeviceCopy> ReadRange for TestRange<T> {
        type Item = T;
    }

    impl<T: DeviceCopy> WriteRange for TestRange<T> {
        type Item = T;
    }

    impl<T> RandomAccessRange for TestRange<T> {}
    impl<T> ContiguousRange for TestRange<T> {}

    fn read_i32<R: ReadRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn write_i32<R: WriteRange<Item = i32> + RandomAccessRange>(_range: R) {}

    #[test]
    fn rust_borrows_supply_range_capabilities() {
        let mut values = TestRange::<i32>::new(3);
        read_i32(&values);
        read_i32(&mut values);
        write_i32(&mut values);
        assert_eq!(DeviceRange::len(&values), 3);
    }

    #[test]
    fn double_buffer_tracks_shape_without_host_data_access() {
        let mut buffers = DoubleBuffer::new(TestRange::<i32>::new(3), TestRange::new(3))
            .expect("equal ranges must construct");
        assert_eq!(buffers.len(), 3);
        assert_eq!(DeviceRange::len(buffers.current()), 3);
        buffers.swap();
        assert_eq!(DeviceRange::len(buffers.alternate_mut()), 3);
        let _parts = buffers.into_parts();
        assert!(DoubleBuffer::new(TestRange::<i32>::new(1), TestRange::<i32>::new(2)).is_none());
    }

    #[test]
    fn double_buffer_split_borrow_follows_the_selected_side() {
        let mut buffers = DoubleBuffer::new(TestRange::<u32>::new(1), TestRange::new(1))
            .expect("equal ranges must construct");
        let (current, alternate) = buffers.current_and_alternate_mut();
        current.tag = 3;
        alternate.tag = 4;
        assert_eq!(buffers.current().tag, 3);

        buffers.swap();
        let (current, alternate) = buffers.current_and_alternate_mut();
        current.tag = 5;
        alternate.tag = 6;
        assert_eq!(buffers.current().tag, 5);
        let (first, second) = buffers.into_parts();
        assert_eq!((first.tag, second.tag), (6, 5));
    }

    #[test]
    fn paired_double_buffer_rejects_every_shape_mismatch() {
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
            )
            .is_some()
        );
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(2)),
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
            )
            .is_none()
        );
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
                (TestRange::<u32>::new(2), TestRange::<i32>::new(3)),
            )
            .is_none()
        );
    }
}
