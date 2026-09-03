include!("support/histogram_scalar_probe.rs");

histogram_scalar_probe!("DeviceHistogram 32-bit scalar matrix"; (i32, i32), (u32, i32), (f32, f32));
