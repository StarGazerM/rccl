include!("support/histogram_scalar_probe.rs");

histogram_scalar_probe!("DeviceHistogram 8-bit scalar matrix"; (i8, i32), (u8, i32));
