use core::marker::PhantomData;

use rccl_memory::ContiguousDeviceRange;

use crate::{MapError, MapKey, MapValue};

/// Borrowed structure-of-arrays key/value input for a bulk map operation.
pub struct Entries<'a, K, V, KR: ?Sized, VR: ?Sized> {
    pub(crate) keys: &'a KR,
    pub(crate) values: &'a VR,
    marker: PhantomData<fn() -> (K, V)>,
}

pub fn entries<'a, K, V, KR, VR>(
    keys: &'a KR,
    values: &'a VR,
) -> Result<Entries<'a, K, V, KR, VR>, MapError>
where
    K: MapKey,
    V: MapValue,
    KR: ContiguousDeviceRange<K> + ?Sized,
    VR: ContiguousDeviceRange<V> + ?Sized,
{
    if keys.len() != values.len() {
        return Err(MapError::LengthMismatch {
            keys: keys.len(),
            values: values.len(),
        });
    }
    Ok(Entries {
        keys,
        values,
        marker: PhantomData,
    })
}

impl<K, V, KR: ?Sized, VR: ?Sized> Entries<'_, K, V, KR, VR> {
    pub fn len(&self) -> usize
    where
        KR: ContiguousDeviceRange<K>,
    {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool
    where
        KR: ContiguousDeviceRange<K>,
    {
        self.keys.len() == 0
    }
}
