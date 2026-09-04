/// Errors from the implemented in-place repeated-value transform slice.
#[derive(Debug)]
pub enum TransformError {
    LengthMismatch { input: usize, output: usize },
    Execution(crate::DeviceAlgorithmError),
}

impl core::fmt::Display for TransformError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { input, output } => {
                write!(
                    formatter,
                    "transform input length {input} differs from output length {output}"
                )
            }
            Self::Execution(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LengthMismatch { .. } => None,
            Self::Execution(error) => Some(error),
        }
    }
}
