//! Context-scoped CUDA module state owned by CUB host dispatch.

use cuda_core::{CudaContext, CudaModule, DriverError};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    error::Error,
    fmt,
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Debug)]
#[non_exhaustive]
pub enum KernelContextError {
    Driver(DriverError),
    Module(cuda_host::EmbeddedModuleError),
}

impl fmt::Display for KernelContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::Module(error) => write!(f, "embedded CUDA module error: {error}"),
        }
    }
}

impl Error for KernelContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Module(error) => Some(error),
        }
    }
}

impl From<DriverError> for KernelContextError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<cuda_host::EmbeddedModuleError> for KernelContextError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        Self::Module(value)
    }
}

struct ContextModules {
    module: Arc<CudaModule>,
    multiprocessors: u32,
    typed: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl ContextModules {
    fn context(&self) -> &Arc<CudaContext> {
        self.module.context()
    }

    fn typed<T, E>(&self, build: impl FnOnce(&Arc<CudaModule>) -> Result<T, E>) -> Result<Arc<T>, E>
    where
        T: Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();
        {
            let typed = self
                .typed
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(existing) = typed.get(&type_id) {
                return Ok(existing
                    .clone()
                    .downcast::<T>()
                    .expect("a TypeId-keyed CUDA module cache entry changed type"));
            }
        }

        let candidate = Arc::new(build(&self.module)?);
        let mut typed = self
            .typed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let selected = typed
            .entry(type_id)
            .or_insert_with(|| candidate.clone())
            .clone();
        Ok(selected
            .downcast::<T>()
            .expect("a TypeId-keyed CUDA module cache entry changed type"))
    }
}

static CONTEXT_MODULES: OnceLock<Mutex<Vec<Weak<ContextModules>>>> = OnceLock::new();

fn context_modules(context: &Arc<CudaContext>) -> Result<Arc<ContextModules>, KernelContextError> {
    let registry = CONTEXT_MODULES.get_or_init(|| Mutex::new(Vec::new()));
    {
        let mut entries = registry.lock().unwrap_or_else(|poison| poison.into_inner());
        entries.retain(|entry| entry.strong_count() != 0);
        if let Some(modules) = entries
            .iter()
            .filter_map(Weak::upgrade)
            .find(|modules| modules.context().as_ref() == context.as_ref())
        {
            return Ok(modules);
        }
    }

    let module = cuda_host::load_all_ptx_bundles_merged(context)?;
    let candidate = Arc::new(ContextModules {
        module,
        multiprocessors: context.multiprocessor_count()?,
        typed: Mutex::new(HashMap::new()),
    });

    let mut entries = registry.lock().unwrap_or_else(|poison| poison.into_inner());
    entries.retain(|entry| entry.strong_count() != 0);
    if let Some(modules) = entries
        .iter()
        .filter_map(Weak::upgrade)
        .find(|modules| modules.context().as_ref() == context.as_ref())
    {
        return Ok(modules);
    }
    entries.push(Arc::downgrade(&candidate));
    Ok(candidate)
}

/// Live CUB kernel modules for one CUDA context.
///
/// Thrust retains this value in its execution policy, but its representation
/// and typed kernel views remain owned by CUB.
#[doc(hidden)]
#[derive(Clone)]
pub struct KernelContext {
    modules: Arc<ContextModules>,
}

impl KernelContext {
    #[doc(hidden)]
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, KernelContextError> {
        Ok(Self {
            modules: context_modules(context)?,
        })
    }

    pub(crate) fn multiprocessors(&self) -> u32 {
        self.modules.multiprocessors
    }

    pub(crate) fn loaded_module<T, E>(
        &self,
        build: impl FnOnce(&Arc<CudaModule>) -> Result<T, E>,
    ) -> Result<Arc<T>, E>
    where
        T: Send + Sync + 'static,
    {
        self.modules.typed(build)
    }
}
