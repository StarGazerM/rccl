use cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, WarpTile};

/// Device scalar that can cross a warp shuffle without exposing its bit
/// representation to the caller.
#[doc(hidden)]
pub trait WarpValue: Copy {
    fn shuffle_down<const N: u32>(tile: &WarpTile<N>, value: Self, delta: u32) -> Self;
}

macro_rules! impl_warp_value_u32 {
    ($($ty:ty),+ $(,)?) => {$ (
        impl WarpValue for $ty {
            #[inline(always)]
            fn shuffle_down<const N: u32>(tile: &WarpTile<N>, value: Self, delta: u32) -> Self {
                tile.shfl_down(value as u32, delta) as Self
            }
        }
    )+ };
}

impl_warp_value_u32!(u8, u16, u32, i8, i16, i32);

macro_rules! impl_warp_value_u64 {
    ($($ty:ty),+ $(,)?) => {$ (
        impl WarpValue for $ty {
            #[inline(always)]
            fn shuffle_down<const N: u32>(tile: &WarpTile<N>, value: Self, delta: u32) -> Self {
                let bits = value as u64;
                let low = tile.shfl_down(bits as u32, delta) as u64;
                let high = tile.shfl_down((bits >> 32) as u32, delta) as u64;
                (low | (high << 32)) as Self
            }
        }
    )+ };
}

impl_warp_value_u64!(u64, i64);

impl WarpValue for f32 {
    #[inline(always)]
    fn shuffle_down<const N: u32>(tile: &WarpTile<N>, value: Self, delta: u32) -> Self {
        tile.shfl_down_f32(value, delta)
    }
}

impl WarpValue for f64 {
    #[inline(always)]
    fn shuffle_down<const N: u32>(tile: &WarpTile<N>, value: Self, delta: u32) -> Self {
        let bits = value.to_bits();
        let low = tile.shfl_down(bits as u32, delta) as u64;
        let high = tile.shfl_down((bits >> 32) as u32, delta) as u64;
        Self::from_bits(low | (high << 32))
    }
}

/// Reduce the active prefix of a logical warp to lane zero.
///
/// `valid_items` may be smaller than `N` for a partial logical warp. Pass `N`
/// when every lane participates.
#[inline(always)]
pub fn warp_reduce<T, const N: u32>(value: T, valid_items: u32, op: impl Fn(T, T) -> T + Copy) -> T
where
    T: WarpValue,
{
    let tile = cuda_device::cooperative_groups::this_thread_block().tiled_partition::<N>();
    let rank = tile.thread_rank();
    let mut aggregate = value;
    let mut offset = 1;
    while offset < N {
        let other = T::shuffle_down(&tile, aggregate, offset);
        if rank < valid_items && rank + offset < valid_items {
            aggregate = op(aggregate, other);
        }
        offset <<= 1;
    }
    aggregate
}

/// Reduce each thread's fixed-size item array and then reduce the logical warp.
#[inline(always)]
pub fn warp_reduce_items<T, const N: u32, const ITEMS: usize>(
    items: [T; ITEMS],
    op: impl Fn(T, T) -> T + Copy,
) -> T
where
    T: WarpValue,
{
    assert!(ITEMS > 0);
    let mut aggregate = items[0];
    let mut item = 1;
    while item < ITEMS {
        aggregate = op(aggregate, items[item]);
        item += 1;
    }
    warp_reduce::<T, N>(aggregate, N, op)
}
