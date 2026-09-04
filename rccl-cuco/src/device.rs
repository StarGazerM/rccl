//! Ordinary device implementations used by the generated CUDA entry points.
//!
//! The caller-side macro only materializes `#[kernel]` wrappers.  Hash-table
//! logic lives here so it is type-checked, reviewed, and tested as normal Rust
//! instead of being hidden inside a procedural-macro token stream.

use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64},
    thread, warp,
};

use crate::{MapKey, MapValue, ProbingScheme, StaticMapReadRef, StaticMapRef};

#[inline(always)]
pub fn clear(index: thread::ThreadIndex<'_>, mut slots: DisjointSlice<u64>, empty_slot: u64) {
    if let Some(slot) = slots.get_mut(index) {
        *slot = empty_slot;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert<K, V, P, const CG: u32, const ASSIGN: bool>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    keys: &[K],
    values: &[V],
    inserted: &[DeviceAtomicU64],
    invalid: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let input_index = global_thread / CG as usize;
    let active = input_index < keys.len();
    let mut fresh = false;
    let reserved = active
        && (keys[input_index] == empty_key
            || keys[input_index] == erased_key
            || values[input_index].to_bits() == empty_value.to_bits());
    if active && !reserved {
        // SAFETY: map construction validates CG and rounds capacity to a
        // non-empty multiple of CG; the bulk kernel assigns one convergent
        // tile per input and receives only the packed atomic table prefix.
        let map = unsafe {
            StaticMapRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing)
        };
        fresh = map.insert_counted::<ASSIGN>(keys[input_index], values[input_index]);
    }

    // Every launched lane reaches these full-warp ballots, including tail
    // tiles. One counter atomic per warp matches cuCollections' counted bulk
    // operation without adding a per-key counter atomic.
    let tile_rank = global_thread % CG as usize;
    let successes = warp::ballot(active && tile_rank == 0 && fresh).count_ones() as u64;
    if warp::lane_id() == 0 && successes != 0 {
        inserted[0].fetch_add(successes, AtomicOrdering::Relaxed);
    }
    let invalid_keys = warp::ballot(active && tile_rank == 0 && reserved).count_ones() as u64;
    if warp::lane_id() == 0 && invalid_keys != 0 {
        invalid[0].fetch_add(invalid_keys, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert_prevalidated<K, V, P, const CG: u32, const SUPPORTS_ERASE: bool>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    keys: &[K],
    values: &[V],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= keys.len() {
        return;
    }
    // SAFETY: the host map validates the cooperative-group and table-extent
    // invariants, and this kernel assigns one convergent tile per input.
    let map = unsafe {
        StaticMapRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing)
    };
    let _ = map.insert_prevalidated::<SUPPORTS_ERASE>(keys[input_index], values[input_index]);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert_range<K, V, P, Input, const CG: u32, const ASSIGN: bool>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    input: Input,
    len: usize,
    inserted: &[DeviceAtomicU64],
    invalid: &[DeviceAtomicU64],
    range_error: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
{
    let input_index = global_thread / CG as usize;
    let active = input_index < len;
    let item = if active {
        // SAFETY: input_index is bounded by the host range extent.
        unsafe { input.read_at(input_index) }
    } else {
        None
    };
    let mut fresh = false;
    let reserved = item.is_some_and(|item| {
        item.key == empty_key
            || item.key == erased_key
            || item.value.to_bits() == empty_value.to_bits()
    });
    if let Some(item) = item {
        if !reserved {
            // SAFETY: map construction validates the cooperative group and
            // table extent; this kernel assigns one convergent tile per item.
            let map = unsafe {
                StaticMapRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing)
            };
            fresh = map.insert_counted::<ASSIGN>(item.key, item.value);
        }
    } else if active && global_thread.is_multiple_of(CG as usize) {
        range_error[0].store(1, AtomicOrdering::Relaxed);
    }

    let tile_rank = global_thread % CG as usize;
    let successes = warp::ballot(active && tile_rank == 0 && fresh).count_ones() as u64;
    if warp::lane_id() == 0 && successes != 0 {
        inserted[0].fetch_add(successes, AtomicOrdering::Relaxed);
    }
    let invalid_items = warp::ballot(active && tile_rank == 0 && reserved).count_ones() as u64;
    if warp::lane_id() == 0 && invalid_items != 0 {
        invalid[0].fetch_add(invalid_items, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert_range_prevalidated<K, V, P, Input, const CG: u32, const SUPPORTS_ERASE: bool>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    input: Input,
    len: usize,
    range_error: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= len {
        return;
    }
    // SAFETY: input_index is bounded by the host range extent.
    let Some(item) = (unsafe { input.read_at(input_index) }) else {
        if global_thread.is_multiple_of(CG as usize) {
            range_error[0].store(1, AtomicOrdering::Relaxed);
        }
        return;
    };
    // SAFETY: map construction validates the cooperative group and table
    // extent; this kernel assigns one convergent tile per item.
    let map = unsafe {
        StaticMapRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing)
    };
    let _ = map.insert_prevalidated::<SUPPORTS_ERASE>(item.key, item.value);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn find<K, V, P, const CG: u32>(
    global_thread: usize,
    slots: &[u64],
    keys: &[K],
    output: &[V::Atomic],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= keys.len() {
        return;
    }
    let map =
        StaticMapReadRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing);
    let value = map.find(keys[input_index]).unwrap_or(empty_value);
    if global_thread.is_multiple_of(CG as usize) {
        V::store_atomic(output, input_index, value);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn contains<K, V, P, const CG: u32>(
    global_thread: usize,
    slots: &[u64],
    keys: &[K],
    output: &[DeviceAtomicU32],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= keys.len() {
        return;
    }
    let map =
        StaticMapReadRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing);
    let present = map.contains(keys[input_index]);
    if global_thread.is_multiple_of(CG as usize) {
        output[input_index].store(present as u32, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn find_range<K, V, P, Input, Output, const CG: u32>(
    global_thread: usize,
    slots: &[u64],
    input: Input,
    output: Output,
    len: usize,
    range_error: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    Input: crate::range::DeviceMapRead<Item = K>,
    Output: crate::range::DeviceMapWrite<Item = V>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= len {
        return;
    }
    // SAFETY: input_index is bounded by the host range extent.
    let key = unsafe { input.read_at(input_index) };
    let value = key.map_or(empty_value, |key| {
        let map = StaticMapReadRef::<K, V, P, CG>::new(
            slots,
            empty_key,
            empty_value,
            erased_key,
            probing,
        );
        map.find(key).unwrap_or(empty_value)
    });
    if global_thread.is_multiple_of(CG as usize) {
        if key.is_none() {
            range_error[0].store(1, AtomicOrdering::Relaxed);
        }
        // SAFETY: output length was checked against len and only tile leaders
        // write, so distinct input indices cannot alias.
        unsafe { output.write_at(input_index, value) };
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn contains_range<K, V, P, Input, Output, const CG: u32>(
    global_thread: usize,
    slots: &[u64],
    input: Input,
    output: Output,
    len: usize,
    range_error: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    Input: crate::range::DeviceMapRead<Item = K>,
    Output: crate::range::DeviceMapWrite<Item = u32>,
{
    let input_index = global_thread / CG as usize;
    if input_index >= len {
        return;
    }
    // SAFETY: input_index is bounded by the host range extent.
    let key = unsafe { input.read_at(input_index) };
    let present = key.is_some_and(|key| {
        let map = StaticMapReadRef::<K, V, P, CG>::new(
            slots,
            empty_key,
            empty_value,
            erased_key,
            probing,
        );
        map.contains(key)
    });
    if global_thread.is_multiple_of(CG as usize) {
        if key.is_none() {
            range_error[0].store(1, AtomicOrdering::Relaxed);
        }
        // SAFETY: output length was checked against len and only tile leaders
        // write, so distinct input indices cannot alias.
        unsafe { output.write_at(input_index, present as u32) };
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn contains_if_range<K, V, P, Input, Stencil, Output, Predicate, const CG: u32>(
    global_thread: usize,
    slots: &[u64],
    input: Input,
    stencil: Stencil,
    output: Output,
    predicate: Predicate,
    len: usize,
    range_error: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    Input: crate::range::DeviceMapRead<Item = K>,
    Stencil: crate::range::DeviceMapRead,
    Output: crate::range::DeviceMapWrite<Item = u32>,
    Predicate: Fn(Stencil::Item) -> bool + Copy,
{
    let input_index = global_thread / CG as usize;
    if input_index >= len {
        return;
    }
    // Read the stencil first. A false predicate deliberately bypasses both
    // the query range and the map lookup.
    let stencil_item = unsafe { stencil.read_at(input_index) };
    let selected = stencil_item.is_some_and(predicate);
    let key = if selected {
        // SAFETY: input and stencil extents were checked equal on the host.
        unsafe { input.read_at(input_index) }
    } else {
        None
    };
    let present = selected
        && key.is_some_and(|key| {
            let map = StaticMapReadRef::<K, V, P, CG>::new(
                slots,
                empty_key,
                empty_value,
                erased_key,
                probing,
            );
            map.contains(key)
        });
    if global_thread.is_multiple_of(CG as usize) {
        if stencil_item.is_none() {
            range_error[0].store(2, AtomicOrdering::Relaxed);
        } else if selected && key.is_none() {
            range_error[0].store(1, AtomicOrdering::Relaxed);
        }
        // SAFETY: output length was checked against len and only tile leaders
        // write, so distinct input indices cannot alias.
        unsafe { output.write_at(input_index, present as u32) };
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn erase<K, V, P, const CG: u32>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    keys: &[K],
    erased: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let input_index = global_thread / CG as usize;
    let active = input_index < keys.len();
    let mut removed = false;
    if active {
        // SAFETY: the host map validates the cooperative-group and table-extent
        // invariants, and this kernel assigns one convergent tile per input.
        let map = unsafe {
            StaticMapRef::<K, V, P, CG>::new(slots, empty_key, empty_value, erased_key, probing)
        };
        removed = map.erase(keys[input_index]);
    }
    let tile_rank = global_thread % CG as usize;
    let successes = warp::ballot(active && tile_rank == 0 && removed).count_ones() as u64;
    if warp::lane_id() == 0 && successes != 0 {
        erased[0].fetch_add(successes, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
pub fn retrieve_all<K, V>(
    index: usize,
    slots: &[u64],
    keys: &[DeviceAtomicU32],
    values: &[V::Atomic],
    count: &[DeviceAtomicU64],
    empty_key: K,
    erased_key: K,
) where
    K: MapKey,
    V: MapValue,
{
    if index >= slots.len() / V::SLOT_WORDS {
        return;
    }
    let word_index = index * V::SLOT_WORDS;
    let control = slots[word_index];
    let key = K::from_bits(control as u32);
    let occupied = if V::SLOT_WORDS == 1 {
        key != empty_key && key != erased_key
    } else {
        key != empty_key && key != erased_key && ((control >> 32) as u32) == 1
    };
    if occupied {
        let position = count[0].fetch_add(1, AtomicOrdering::Relaxed) as usize;
        keys[position].store(key.to_bits(), AtomicOrdering::Relaxed);
        let value_bits = if V::SLOT_WORDS == 1 {
            (control >> 32) & 0xffff_ffff
        } else {
            slots[word_index + 1]
        };
        V::store_atomic(values, position, V::from_bits(value_bits));
    }
}

#[inline(always)]
pub fn count<K, V>(
    index: usize,
    slots: &[u64],
    count: &[DeviceAtomicU64],
    empty_key: K,
    erased_key: K,
) where
    K: MapKey,
    V: MapValue,
{
    let occupied = if index < slots.len() / V::SLOT_WORDS {
        let control = slots[index * V::SLOT_WORDS];
        let key = K::from_bits(control as u32);
        if V::SLOT_WORDS == 1 {
            key != empty_key && key != erased_key
        } else {
            key != empty_key && key != erased_key && ((control >> 32) as u32) == 1
        }
    } else {
        false
    };
    let occupied_in_warp = warp::ballot(occupied).count_ones() as u64;
    if warp::lane_id() == 0 && occupied_in_warp != 0 {
        count[0].fetch_add(occupied_in_warp, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn rehash<K, V, P, const CG: u32>(
    global_thread: usize,
    source: &[u64],
    destination: &[DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
) where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    let source_index = global_thread / CG as usize;
    if source_index >= source.len() / V::SLOT_WORDS {
        return;
    }
    let word_index = source_index * V::SLOT_WORDS;
    let control = source[word_index];
    let key = K::from_bits(control as u32);
    let occupied = if V::SLOT_WORDS == 1 {
        key != empty_key && key != erased_key
    } else {
        key != empty_key && key != erased_key && ((control >> 32) as u32) == 1
    };
    if !occupied {
        return;
    }
    let value_bits = if V::SLOT_WORDS == 1 {
        (control >> 32) & 0xffff_ffff
    } else {
        source[word_index + 1]
    };
    // SAFETY: both capacities come from validated host maps and this kernel
    // assigns one convergent cooperative tile to each source slot.
    let map = unsafe {
        StaticMapRef::<K, V, P, CG>::new(destination, empty_key, empty_value, erased_key, probing)
    };
    let _ = map.insert(key, V::from_bits(value_bits));
}
