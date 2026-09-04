use std::{error::Error, fmt};

use cuda_core::{DriverError, LaunchContractError};
use rccl_memory::MemoryError;

#[derive(Debug)]
pub enum MapError {
    Memory(MemoryError),
    Launch(DriverError),
    Contract(LaunchContractError),
    InvalidLoadFactor(f64),
    InvalidGrowthFactor(f64),
    CapacityOverflow,
    InsufficientCapacity { entries: usize, capacity: usize },
    InvalidCooperativeGroup(u32),
    EmptyKeyEqualsErasedKey,
    EraseDisabled,
    ReservedSentinels { count: usize },
    LengthMismatch { keys: usize, values: usize },
    OutputTooSmall { needed: usize, available: usize },
    ContextMismatch,
}

impl fmt::Display for MapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(error) => error.fmt(formatter),
            Self::Launch(error) => write!(formatter, "CUDA map kernel launch failed: {error}"),
            Self::Contract(error) => write!(formatter, "CUDA map launch contract failed: {error}"),
            Self::InvalidLoadFactor(value) => write!(
                formatter,
                "map load factor must be in (0, 1], received {value}"
            ),
            Self::InvalidGrowthFactor(value) => write!(
                formatter,
                "dynamic-map growth factor must be finite and greater than 1, received {value}"
            ),
            Self::CapacityOverflow => formatter.write_str("static-map capacity overflowed"),
            Self::InsufficientCapacity { entries, capacity } => write!(
                formatter,
                "cannot rehash {entries} entries into capacity {capacity}"
            ),
            Self::InvalidCooperativeGroup(size) => write!(
                formatter,
                "cooperative group size must be a power of two in 1..=32 and divide 128; received {size}"
            ),
            Self::EmptyKeyEqualsErasedKey => {
                formatter.write_str("empty and erased key sentinels must differ")
            }
            Self::EraseDisabled => formatter.write_str(
                "erase requires a distinct erased-key sentinel in the map configuration",
            ),
            Self::ReservedSentinels { count } => write!(
                formatter,
                "input contains {count} reserved sentinel key/value pair(s); other entries may already be inserted"
            ),
            Self::LengthMismatch { keys, values } => write!(
                formatter,
                "map key/value input lengths differ: {keys} and {values}"
            ),
            Self::OutputTooSmall { needed, available } => write!(
                formatter,
                "map output needs {needed} elements but only {available} are available"
            ),
            Self::ContextMismatch => {
                formatter.write_str("map operands and execution must share one CUDA context")
            }
        }
    }
}

impl Error for MapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Memory(error) => Some(error),
            Self::Launch(error) => Some(error),
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MemoryError> for MapError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<DriverError> for MapError {
    fn from(value: DriverError) -> Self {
        Self::Launch(value)
    }
}

impl From<LaunchContractError> for MapError {
    fn from(value: LaunchContractError) -> Self {
        Self::Contract(value)
    }
}
