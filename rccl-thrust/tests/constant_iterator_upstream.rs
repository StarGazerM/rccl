use rccl_thrust::{DeviceRange, RandomAccessRange, ReadRange, range::RepeatN};

fn require_input_random_access<T>()
where
    T: ReadRange<Item = i32> + RandomAccessRange,
{
}

#[test]
fn pinned_constant_iterator_example_has_exact_output() {
    let input = [3_i32, 7, 2, 5];
    let repeated = RepeatN::new(10_i32, input.len());
    let expected = input.map(|value| value + *repeated.value());
    assert_eq!(expected, [13, 17, 12, 15]);
    assert_eq!(repeated.len(), input.len());
}

#[test]
fn repeat_n_is_copy_and_keeps_value_and_extent() {
    fn copy<T: Copy>(value: T) -> (T, T) {
        (value, value)
    }
    let (first, second) = copy(RepeatN::new(-7_i64, 1_000_003));
    assert_eq!(*first.value(), -7);
    assert_eq!(first.len(), 1_000_003);
    assert_eq!(*second.value(), -7);
    assert_eq!(second.len(), 1_000_003);
}

#[test]
fn pinned_constant_iterator_traits_map_to_input_random_access_range() {
    // Pinned `TestConstantIteratorTraits`: Rust uses capabilities instead of
    // recreating iterator category and traversal-tag TMP.
    require_input_random_access::<RepeatN<i32>>();
}

#[test]
fn pinned_big_distance_is_a_full_width_finite_extent() {
    // Pinned `TestConstantIteratorIncrementBig` checks a distance of 10^10.
    let repeated = RepeatN::new(1_i64, 10_000_000_000_usize);
    assert_eq!(repeated.len(), 10_000_000_000);
}
