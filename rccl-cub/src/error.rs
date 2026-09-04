/// Failure reported by a device-wide operation.
#[derive(Debug)]
pub enum DeviceAlgorithmError {
    LengthMismatch {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    WorkspaceTooSmall {
        role: &'static str,
        required: usize,
        actual: usize,
    },
    InvalidBitRange {
        begin: u32,
        end: u32,
        key_bits: u32,
    },
    InvalidHistogramShape(&'static str),
    UnsupportedTarget(&'static str),
    SizeOverflow,
    Memory(rccl_memory::MemoryError),
    Driver(cuda_core::DriverError),
    Launch(cuda_core::LaunchContractError),
}

impl core::fmt::Display for DeviceAlgorithmError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} length {actual} does not match required length {expected}"
            ),
            Self::WorkspaceTooSmall {
                role,
                required,
                actual,
            } => write!(
                formatter,
                "{role} length {actual} is smaller than required length {required}"
            ),
            Self::InvalidBitRange {
                begin,
                end,
                key_bits,
            } => write!(
                formatter,
                "radix bit range {begin}..{end} is outside 0..={key_bits} or reversed"
            ),
            Self::InvalidHistogramShape(message) => formatter.write_str(message),
            Self::UnsupportedTarget(message) => formatter.write_str(message),
            Self::SizeOverflow => formatter.write_str("combined device range length overflowed"),
            Self::Memory(error) => error.fmt(formatter),
            Self::Driver(error) => error.fmt(formatter),
            Self::Launch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DeviceAlgorithmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::Launch(error) => Some(error),
            Self::LengthMismatch { .. }
            | Self::WorkspaceTooSmall { .. }
            | Self::InvalidBitRange { .. }
            | Self::InvalidHistogramShape(_)
            | Self::UnsupportedTarget(_)
            | Self::SizeOverflow => None,
        }
    }
}

impl From<cuda_core::DriverError> for DeviceAlgorithmError {
    fn from(value: cuda_core::DriverError) -> Self {
        Self::Driver(value)
    }
}
