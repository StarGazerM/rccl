//! GPU result tests ported from pinned CCCL's `thrust/testing/set_*.cu`
//! and `set_*_by_key*.cu` families.

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};
use rccl_memory::TemporaryValue;
use rccl_thrust::KeyValue;
use rccl_thrust::prelude::*;
use std::cmp::Ordering;
use std::error::Error;

fn prefix<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    stream: &cuda_core::CudaStream,
    len: usize,
) -> Result<Vec<T>, cuda_core::DriverError> {
    let values = buffer.to_host_vec(stream)?;
    Ok(values[..len].to_vec())
}

#[derive(Clone, Copy)]
enum Operation {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

fn host_set<T: Copy + PartialOrd>(first: &[T], second: &[T], operation: Operation) -> Vec<T> {
    host_set_by(first, second, operation, |left, right| {
        if left < right {
            Ordering::Less
        } else if right < left {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    })
}

fn host_set_by<T: Copy, F: Fn(&T, &T) -> Ordering + Copy>(
    first: &[T],
    second: &[T],
    operation: Operation,
    order: F,
) -> Vec<T> {
    let mut output = Vec::with_capacity(first.len() + second.len());
    let (mut a, mut b) = (0usize, 0usize);
    while a < first.len() && b < second.len() {
        let first_before = order(&first[a], &second[b]) == Ordering::Less;
        let second_before = order(&second[b], &first[a]) == Ordering::Less;
        match operation {
            Operation::Union => {
                if second_before {
                    output.push(second[b]);
                    b += 1;
                } else {
                    output.push(first[a]);
                    a += 1;
                    if !first_before {
                        b += 1;
                    }
                }
            }
            Operation::Intersection => {
                if first_before {
                    a += 1;
                } else if second_before {
                    b += 1;
                } else {
                    output.push(first[a]);
                    a += 1;
                    b += 1;
                }
            }
            Operation::Difference => {
                if first_before {
                    output.push(first[a]);
                    a += 1;
                } else if second_before {
                    b += 1;
                } else {
                    a += 1;
                    b += 1;
                }
            }
            Operation::SymmetricDifference => {
                if first_before {
                    output.push(first[a]);
                    a += 1;
                } else if second_before {
                    output.push(second[b]);
                    b += 1;
                } else {
                    a += 1;
                    b += 1;
                }
            }
        }
    }
    if matches!(operation, Operation::Union | Operation::SymmetricDifference) {
        output.extend_from_slice(&first[a..]);
        output.extend_from_slice(&second[b..]);
    } else if matches!(operation, Operation::Difference) {
        output.extend_from_slice(&first[a..]);
    }
    output
}

trait TestScalar: TemporaryValue + Copy + std::fmt::Debug + PartialEq + PartialOrd + 'static {
    fn from_i32(value: i32) -> Self;
}

macro_rules! test_scalar {
    ($($ty:ty),+ $(,)?) => {$(
        impl TestScalar for $ty {
            fn from_i32(value: i32) -> Self {
                value as Self
            }
        }
    )+};
}

test_scalar!(i8, u8, i16, u16, i32, u32, f32, f64);

fn test_key_value_type<T, A>(
    execution: &rccl_thrust::cuda::Execution<'_, '_, A>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>>
where
    T: TestScalar,
    A: rccl_thrust::memory::DeviceAllocator,
{
    let mut first = (0..4097_i32)
        .map(|index| KeyValue {
            key: T::from_i32((index * 37 + 11) % 17),
            value: T::from_i32((index * 5 + 1) % 97),
        })
        .collect::<Vec<_>>();
    let mut second = (0..4097_i32)
        .map(|index| KeyValue {
            key: T::from_i32((index * 29 + 7) % 17),
            value: T::from_i32((index * 11 + 3) % 97),
        })
        .collect::<Vec<_>>();
    let order = |left: &KeyValue<T, T>, right: &KeyValue<T, T>| {
        if left.key < right.key {
            Ordering::Less
        } else if right.key < left.key {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    };
    first.sort_by(order);
    second.sort_by(order);
    let device_first = DeviceBuffer::from_host(stream, &first)?;
    let device_second = DeviceBuffer::from_host(stream, &second)?;

    for operation in [
        Operation::Union,
        Operation::Intersection,
        Operation::Difference,
        Operation::SymmetricDifference,
    ] {
        let expected = host_set_by(&first, &second, operation, order);
        let mut output = DeviceBuffer::<KeyValue<T, T>>::zeroed(stream, expected.len())?;
        let len = match operation {
            Operation::Union => device_first
                .view()
                .union_by(device_second.view(), order)
                .write_into(output.view_mut(), execution)?,
            Operation::Intersection => device_first
                .view()
                .intersection_by(device_second.view(), order)
                .write_into(output.view_mut(), execution)?,
            Operation::Difference => device_first
                .view()
                .difference_by(device_second.view(), order)
                .write_into(output.view_mut(), execution)?,
            Operation::SymmetricDifference => device_first
                .view()
                .symmetric_difference_by(device_second.view(), order)
                .write_into(output.view_mut(), execution)?,
        };
        assert_eq!(prefix(&output, stream, len)?, expected);
    }

    first.sort_by(|left, right| order(right, left));
    second.sort_by(|left, right| order(right, left));
    let device_first = DeviceBuffer::from_host(stream, &first)?;
    let device_second = DeviceBuffer::from_host(stream, &second)?;
    let descending = move |left: &KeyValue<T, T>, right: &KeyValue<T, T>| order(right, left);
    let expected = host_set_by(&first, &second, Operation::Union, descending);
    let mut output = DeviceBuffer::<KeyValue<T, T>>::zeroed(stream, expected.len())?;
    let len = device_first
        .view()
        .union_by(device_second.view(), descending)
        .write_into(output.view_mut(), execution)?;
    assert_eq!(prefix(&output, stream, len)?, expected);
    Ok(())
}

fn test_by_key_type<T, A>(
    execution: &rccl_thrust::cuda::Execution<'_, '_, A>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>>
where
    T: TestScalar,
    A: rccl_thrust::memory::DeviceAllocator,
{
    let mut first = (0..4097_i32)
        .map(|index| KeyValue {
            key: T::from_i32((index * 37 + 11) % 17),
            value: T::from_i32((index * 5 + 1) % 97),
        })
        .collect::<Vec<_>>();
    let mut second = (0..8194_i32)
        .map(|index| KeyValue {
            key: T::from_i32((index * 29 + 7) % 17),
            value: T::from_i32((index * 11 + 3) % 97),
        })
        .collect::<Vec<_>>();
    let ascending = |left: &KeyValue<T, T>, right: &KeyValue<T, T>| {
        if left.key < right.key {
            Ordering::Less
        } else if right.key < left.key {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    };

    for descending in [false, true] {
        if descending {
            first.sort_by(|left, right| ascending(right, left));
            second.sort_by(|left, right| ascending(right, left));
        } else {
            first.sort_by(ascending);
            second.sort_by(ascending);
        }
        let order = move |left: &T, right: &T| {
            let natural = if left < right {
                Ordering::Less
            } else if right < left {
                Ordering::Greater
            } else {
                Ordering::Equal
            };
            if descending {
                natural.reverse()
            } else {
                natural
            }
        };
        let record_order =
            |left: &KeyValue<T, T>, right: &KeyValue<T, T>| order(&left.key, &right.key);
        let first_keys_host = first.iter().map(|item| item.key).collect::<Vec<_>>();
        let first_values_host = first.iter().map(|item| item.value).collect::<Vec<_>>();
        let second_keys_host = second.iter().map(|item| item.key).collect::<Vec<_>>();
        let second_values_host = second.iter().map(|item| item.value).collect::<Vec<_>>();
        let first_keys = DeviceBuffer::from_host(stream, &first_keys_host)?;
        let first_values = DeviceBuffer::from_host(stream, &first_values_host)?;
        let second_keys = DeviceBuffer::from_host(stream, &second_keys_host)?;
        let second_values = DeviceBuffer::from_host(stream, &second_values_host)?;
        let first_range = first_keys.view().with_values(first_values.view())?;
        let second_range = second_keys.view().with_values(second_values.view())?;

        for operation in [
            Operation::Union,
            Operation::Intersection,
            Operation::Difference,
            Operation::SymmetricDifference,
        ] {
            let expected = host_set_by(&first, &second, operation, record_order);
            let mut output_keys = DeviceBuffer::<T>::zeroed(stream, expected.len())?;
            let mut output_values = DeviceBuffer::<T>::zeroed(stream, expected.len())?;
            let mut output = output_keys
                .view_mut()
                .with_values(output_values.view_mut())?;
            let len = match operation {
                Operation::Union => first_range
                    .union_by(second_range, order)
                    .write_into(output.reborrow(), execution)?,
                Operation::Intersection => first_range
                    .intersection_by(second_keys.view(), order)
                    .write_into(output.reborrow(), execution)?,
                Operation::Difference => first_range
                    .difference_by(second_range, order)
                    .write_into(output.reborrow(), execution)?,
                Operation::SymmetricDifference => first_range
                    .symmetric_difference_by(second_range, order)
                    .write_into(output.reborrow(), execution)?,
            };
            let actual_keys = prefix(&output_keys, stream, len)?;
            let actual_values = prefix(&output_values, stream, len)?;
            let actual = actual_keys
                .into_iter()
                .zip(actual_values)
                .map(|(key, value)| KeyValue { key, value })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let exec = thrust.par(&stream)?;

    // Exact simple cases from the four key-only CCCL unit-test files.
    let union_a = DeviceBuffer::from_host(&stream, &[0_i32, 2, 2])?;
    let union_b = DeviceBuffer::from_host(&stream, &[0_i32, 2, 2, 2, 3])?;
    let mut union_out = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let union_len = union_a
        .view()
        .union(union_b.view())
        .write_into(union_out.view_mut(), &exec)?;
    assert_eq!(prefix(&union_out, &stream, union_len)?, [0, 2, 2, 2, 3]);

    let first = DeviceBuffer::from_host(&stream, &[0_i32, 2, 4, 6])?;
    let second = DeviceBuffer::from_host(&stream, &[0_i32, 3, 3, 4, 7])?;
    let mut intersection = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let intersection_len = first
        .view()
        .intersection(second.view())
        .write_into(intersection.view_mut(), &exec)?;
    assert_eq!(prefix(&intersection, &stream, intersection_len)?, [0, 4]);

    let mut difference = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let difference_len = first
        .view()
        .difference(second.view())
        .write_into(difference.view_mut(), &exec)?;
    assert_eq!(prefix(&difference, &stream, difference_len)?, [2, 6]);

    let mut symmetric = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let symmetric_len = first
        .view()
        .symmetric_difference(second.view())
        .write_into(symmetric.view_mut(), &exec)?;
    assert_eq!(prefix(&symmetric, &stream, symmetric_len)?, [2, 3, 3, 6, 7]);

    // The descending overload families use the same inferred Rust closure
    // shape as `cuda::std::greater<T>` in the pinned tests.
    let descending_first = DeviceBuffer::from_host(&stream, &[6_i32, 4, 2, 0])?;
    let descending_second = DeviceBuffer::from_host(&stream, &[7_i32, 4, 3, 3, 0])?;
    let descending = |left: &i32, right: &i32| right.cmp(left);
    let mut descending_union = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let len = descending_first
        .view()
        .union_by(descending_second.view(), descending)
        .write_into(descending_union.view_mut(), &exec)?;
    assert_eq!(
        prefix(&descending_union, &stream, len)?,
        [7, 6, 4, 3, 3, 2, 0]
    );

    let mut descending_intersection = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let len = descending_first
        .view()
        .intersection_by(descending_second.view(), descending)
        .write_into(descending_intersection.view_mut(), &exec)?;
    assert_eq!(prefix(&descending_intersection, &stream, len)?, [4, 0]);

    let mut descending_difference = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let len = descending_first
        .view()
        .difference_by(descending_second.view(), descending)
        .write_into(descending_difference.view_mut(), &exec)?;
    assert_eq!(prefix(&descending_difference, &stream, len)?, [6, 2]);

    let mut descending_symmetric = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let len = descending_first
        .view()
        .symmetric_difference_by(descending_second.view(), descending)
        .write_into(descending_symmetric.view_mut(), &exec)?;
    assert_eq!(
        prefix(&descending_symmetric, &stream, len)?,
        [7, 6, 3, 3, 2]
    );

    // Exact by-key simple cases. Equal keys are left-biased and intersection
    // values come only from the first range.
    let a_keys = DeviceBuffer::from_host(&stream, &[0_i32, 2, 4, 6])?;
    let b_keys = DeviceBuffer::from_host(&stream, &[0_i32, 3, 3, 4, 7])?;
    let a_values = DeviceBuffer::from_host(&stream, &[10_i32, 12, 14, 16])?;
    let b_values = DeviceBuffer::from_host(&stream, &[20_i32, 23, 23, 24, 27])?;

    let mut out_keys = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut out_values = DeviceBuffer::<i32>::zeroed(&stream, out_keys.len())?;
    let len = a_keys
        .view()
        .with_values(a_values.view())?
        .union(b_keys.view().with_values(b_values.view())?)
        .write_into(
            out_keys.view_mut().with_values(out_values.view_mut())?,
            &exec,
        )?;
    assert_eq!(prefix(&out_keys, &stream, len)?, [0, 2, 3, 3, 4, 6, 7]);
    assert_eq!(
        prefix(&out_values, &stream, len)?,
        [10, 12, 23, 23, 14, 16, 27]
    );

    let mut out_keys = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let mut out_values = DeviceBuffer::<i32>::zeroed(&stream, out_keys.len())?;
    let len = a_keys
        .view()
        .with_values(a_values.view())?
        .intersection(b_keys.view())
        .write_into(
            out_keys.view_mut().with_values(out_values.view_mut())?,
            &exec,
        )?;
    assert_eq!(prefix(&out_keys, &stream, len)?, [0, 4]);
    assert_eq!(prefix(&out_values, &stream, len)?, [10, 14]);

    let mut out_keys = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let mut out_values = DeviceBuffer::<i32>::zeroed(&stream, out_keys.len())?;
    let len = a_keys
        .view()
        .with_values(a_values.view())?
        .difference(b_keys.view().with_values(b_values.view())?)
        .write_into(
            out_keys.view_mut().with_values(out_values.view_mut())?,
            &exec,
        )?;
    assert_eq!(prefix(&out_keys, &stream, len)?, [2, 6]);
    assert_eq!(prefix(&out_values, &stream, len)?, [12, 16]);

    let mut out_keys = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let mut out_values = DeviceBuffer::<i32>::zeroed(&stream, out_keys.len())?;
    let len = a_keys
        .view()
        .with_values(a_values.view())?
        .symmetric_difference(b_keys.view().with_values(b_values.view())?)
        .write_into(
            out_keys.view_mut().with_values(out_values.view_mut())?,
            &exec,
        )?;
    assert_eq!(prefix(&out_keys, &stream, len)?, [2, 3, 3, 6, 7]);
    assert_eq!(prefix(&out_values, &stream, len)?, [12, 23, 23, 16, 27]);

    // The four pinned descending by-key files use the same comparator and
    // require the same left-biased source-value selection as the ascending
    // overloads.
    let descending_a_values = DeviceBuffer::from_host(&stream, &[16_i32, 14, 12, 10])?;
    let descending_b_values = DeviceBuffer::from_host(&stream, &[27_i32, 24, 23, 33, 20])?;
    let descending_a = descending_first
        .view()
        .with_values(descending_a_values.view())?;
    let descending_b = descending_second
        .view()
        .with_values(descending_b_values.view())?;

    let mut keys = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut values = DeviceBuffer::<i32>::zeroed(&stream, keys.len())?;
    let len = descending_a
        .union_by(descending_b, descending)
        .write_into(keys.view_mut().with_values(values.view_mut())?, &exec)?;
    assert_eq!(prefix(&keys, &stream, len)?, [7, 6, 4, 3, 3, 2, 0]);
    assert_eq!(prefix(&values, &stream, len)?, [27, 16, 14, 23, 33, 12, 10]);

    let mut keys = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let mut values = DeviceBuffer::<i32>::zeroed(&stream, keys.len())?;
    let len = descending_a
        .intersection_by(descending_second.view(), descending)
        .write_into(keys.view_mut().with_values(values.view_mut())?, &exec)?;
    assert_eq!(prefix(&keys, &stream, len)?, [4, 0]);
    assert_eq!(prefix(&values, &stream, len)?, [14, 10]);

    let mut keys = DeviceBuffer::<i32>::zeroed(&stream, 2)?;
    let mut values = DeviceBuffer::<i32>::zeroed(&stream, keys.len())?;
    let len = descending_a
        .difference_by(descending_b, descending)
        .write_into(keys.view_mut().with_values(values.view_mut())?, &exec)?;
    assert_eq!(prefix(&keys, &stream, len)?, [6, 2]);
    assert_eq!(prefix(&values, &stream, len)?, [16, 12]);

    let mut keys = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let mut values = DeviceBuffer::<i32>::zeroed(&stream, keys.len())?;
    let len = descending_a
        .symmetric_difference_by(descending_b, descending)
        .write_into(keys.view_mut().with_values(values.view_mut())?, &exec)?;
    assert_eq!(prefix(&keys, &stream, len)?, [7, 6, 3, 3, 2]);
    assert_eq!(prefix(&values, &stream, len)?, [27, 16, 23, 33, 12]);

    // Multi-tile tests exercise balanced partitions and decoupled lookback.
    let n = 25_000usize;
    let even_host: Vec<i32> = (0..n).map(|value| (value * 2) as i32).collect();
    let odd_host: Vec<i32> = (0..n).map(|value| (value * 2 + 1) as i32).collect();
    let evens = DeviceBuffer::from_host(&stream, &even_host)?;
    let odds = DeviceBuffer::from_host(&stream, &odd_host)?;
    let mut large = DeviceBuffer::<i32>::zeroed(&stream, 2 * n)?;
    let len = evens
        .view()
        .union(odds.view())
        .write_into(large.view_mut(), &exec)?;
    assert_eq!(len, 2 * n);
    assert_eq!(
        prefix(&large, &stream, len)?,
        (0..2 * n as i32).collect::<Vec<_>>()
    );

    let mut empty_intersection = DeviceBuffer::<i32>::zeroed(&stream, n)?;
    assert_eq!(
        evens
            .view()
            .intersection(odds.view())
            .write_into(empty_intersection.view_mut(), &exec)?,
        0
    );

    // Direct ports of the pinned discard_iterator tests: the Rust output
    // range has a finite logical extent, but the generated set agent removes
    // only the final global write and returns the same output position.
    assert_eq!(evens.view().union(odds.view()).count(&exec)?, 2 * n);
    assert_eq!(evens.view().intersection(odds.view()).count(&exec)?, 0);

    let equal = DeviceBuffer::from_host(&stream, &even_host)?;
    let mut equal_union = DeviceBuffer::<i32>::zeroed(&stream, 2 * n)?;
    let len = evens
        .view()
        .union(equal.view())
        .write_into(equal_union.view_mut(), &exec)?;
    assert_eq!(len, n);
    assert_eq!(prefix(&equal_union, &stream, len)?, even_host);

    let mut equal_difference = DeviceBuffer::<i32>::zeroed(&stream, n)?;
    assert_eq!(
        evens
            .view()
            .difference(equal.view())
            .write_into(equal_difference.view_mut(), &exec)?,
        0
    );

    // Port the variable-size and low-cardinality multiset structure used by
    // all four pinned key-only test files. The host side is the reference
    // system in those files; every result is compared, not merely sortedness.
    let first_host = (0..4097)
        .map(|index| ((index * 37 + 11) % 13) - 6)
        .collect::<Vec<_>>();
    let second_host = (0..8194)
        .map(|index| ((index * 29 + 7) % 13) - 6)
        .collect::<Vec<_>>();
    let mut first_host = first_host;
    let mut second_host = second_host;
    first_host.sort();
    second_host.sort();
    let first = DeviceBuffer::from_host(&stream, &first_host)?;
    for second_len in [0usize, 1, 2048, 4097, 4098, 8194] {
        let second_part = &second_host[..second_len];
        let second = DeviceBuffer::from_host(&stream, second_part)?;

        let expected = host_set(&first_host, second_part, Operation::Union);
        let mut output = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
        let len = first
            .view()
            .union(second.view())
            .write_into(output.view_mut(), &exec)?;
        assert_eq!(prefix(&output, &stream, len)?, expected);

        let expected = host_set(&first_host, second_part, Operation::Intersection);
        let mut output = DeviceBuffer::<i32>::zeroed(&stream, first.len().min(second.len()))?;
        let len = first
            .view()
            .intersection(second.view())
            .write_into(output.view_mut(), &exec)?;
        assert_eq!(prefix(&output, &stream, len)?, expected);

        let expected = host_set(&first_host, second_part, Operation::Difference);
        let mut output = DeviceBuffer::<i32>::zeroed(&stream, first.len())?;
        let len = first
            .view()
            .difference(second.view())
            .write_into(output.view_mut(), &exec)?;
        assert_eq!(prefix(&output, &stream, len)?, expected);

        let expected = host_set(&first_host, second_part, Operation::SymmetricDifference);
        let mut output = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
        let len = first
            .view()
            .symmetric_difference(second.view())
            .write_into(output.view_mut(), &exec)?;
        assert_eq!(prefix(&output, &stream, len)?, expected);
    }

    let empty = DeviceBuffer::<i32>::from_host(&stream, &[])?;
    let mut empty_output = DeviceBuffer::<i32>::from_host(&stream, &[])?;
    assert_eq!(
        empty
            .view()
            .union(empty.view())
            .write_into(empty_output.view_mut(), &exec)?,
        0
    );

    // Direct Rust counterpart of every pinned `key_value<U, U>` instantiation.
    // CCCL orders only by `key` but compares whole records for test equality;
    // Rust exposes that key equivalence as a closure rather than a dishonest
    // `Ord` implementation.
    test_key_value_type::<i8, _>(&exec, &stream)?;
    test_key_value_type::<u8, _>(&exec, &stream)?;
    test_key_value_type::<i16, _>(&exec, &stream)?;
    test_key_value_type::<u16, _>(&exec, &stream)?;
    test_key_value_type::<i32, _>(&exec, &stream)?;
    test_key_value_type::<u32, _>(&exec, &stream)?;
    test_key_value_type::<f32, _>(&exec, &stream)?;
    test_key_value_type::<f64, _>(&exec, &stream)?;

    // The pinned by-key families instantiate both keys and values at every
    // variable-unit-test type, for ascending and descending overloads.
    test_by_key_type::<i8, _>(&exec, &stream)?;
    test_by_key_type::<u8, _>(&exec, &stream)?;
    test_by_key_type::<i16, _>(&exec, &stream)?;
    test_by_key_type::<u16, _>(&exec, &stream)?;
    test_by_key_type::<i32, _>(&exec, &stream)?;
    test_by_key_type::<u32, _>(&exec, &stream)?;
    test_by_key_type::<f32, _>(&exec, &stream)?;
    test_by_key_type::<f64, _>(&exec, &stream)?;

    // Exercise the arithmetic-range lowering under both normal execution and
    // compute-sanitizer without materializing either input.
    let first = rccl_thrust::arithmetic_range(0_i64, 1_i64, 10_001);
    let second = rccl_thrust::arithmetic_range(0_i64, 1_i64, 10_000);
    let mut output = DeviceBuffer::<i64>::zeroed(&stream, 1)?;
    let len = first
        .difference(second)
        .write_into(output.view_mut(), &exec)?;
    assert_eq!(len, 1);
    assert_eq!(prefix(&output, &stream, len)?, [10_000]);

    if !std::env::args().any(|argument| argument == "--skip-wide-index") {
        // Exact Rust range equivalent of the pinned counting_iterator tests.
        // Magnitudes 30..33 cross CCCL's signed-32-bit double-index threshold
        // and exercise the i64/u128 descriptor path. The sanitizer gate skips
        // only these impractically large launches; this native result gate is
        // mandatory and the same arithmetic agent is sanitized above.
        for magnitude in 30..=33 {
            let count = 1usize << magnitude;
            let first = rccl_thrust::arithmetic_range(0_i64, 1_i64, count + 1);
            let second = rccl_thrust::arithmetic_range(0_i64, 1_i64, count);
            let len = first
                .difference(second)
                .write_into(output.view_mut(), &exec)?;
            assert_eq!(len, 1);
            assert_eq!(prefix(&output, &stream, len)?, [count as i64]);

            let first = rccl_thrust::arithmetic_range(0_i64, 1_i64, count + 1);
            let second = rccl_thrust::arithmetic_range(count as i64, 1_i64, count);
            let len = first
                .intersection(second)
                .write_into(output.view_mut(), &exec)?;
            assert_eq!(len, 1);
            assert_eq!(prefix(&output, &stream, len)?, [count as i64]);
        }
    }

    println!("pinned CCCL Thrust set-operation GPU tests: PASS");
    Ok(())
}
