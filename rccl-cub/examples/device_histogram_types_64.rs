include!("support/histogram_scalar_probe.rs");

histogram_scalar_probe!("DeviceHistogram 64-bit scalar matrix"; (i64, i32), (u64, i32), (f64, f64));
