use crate::{CudaAsyncMemoryResource, DeviceAllocator, MemoryError};
use cuda_core::CudaContext;
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
};

/// Explicitly erased, thread-safe allocator handle used only by the optional
/// process-wide compatibility registry.
pub type SharedDeviceAllocator = Arc<dyn DeviceAllocator + Send + Sync>;

type ResourceMap = HashMap<usize, SharedDeviceAllocator>;

fn resources() -> &'static RwLock<ResourceMap> {
    static RESOURCES: OnceLock<RwLock<ResourceMap>> = OnceLock::new();
    RESOURCES.get_or_init(|| RwLock::new(HashMap::new()))
}

fn default_resource(context: Arc<CudaContext>) -> SharedDeviceAllocator {
    Arc::new(CudaAsyncMemoryResource::new(context))
}

/// Returns the process-wide resource selected for `context`'s device.
///
/// The first lookup installs a CUDA stream-ordered resource, matching RMM's
/// per-device default-resource model. The returned `Arc` owns the resource, so
/// replacing the registry entry cannot invalidate existing allocations.
pub fn get_per_device_resource(context: Arc<CudaContext>) -> SharedDeviceAllocator {
    let ordinal = context.ordinal();
    if let Some(resource) = resources()
        .read()
        .expect("per-device resource registry poisoned")
        .get(&ordinal)
        .cloned()
    {
        return resource;
    }

    let mut map = resources()
        .write()
        .expect("per-device resource registry poisoned");
    map.entry(ordinal)
        .or_insert_with(|| default_resource(context))
        .clone()
}

/// Replaces a device's process-wide resource and returns the previous one.
///
/// The resource must belong to the same retained CUDA primary context as
/// `context`. Callers should restore the returned resource after a scoped
/// override so unrelated libraries continue using their expected policy.
pub fn set_per_device_resource(
    context: Arc<CudaContext>,
    resource: SharedDeviceAllocator,
) -> Result<SharedDeviceAllocator, MemoryError> {
    if context.as_ref() != resource.context().as_ref() {
        return Err(MemoryError::ContextMismatch);
    }
    let ordinal = context.ordinal();
    let mut map = resources()
        .write()
        .expect("per-device resource registry poisoned");
    Ok(map
        .insert(ordinal, resource)
        .unwrap_or_else(|| default_resource(context)))
}
