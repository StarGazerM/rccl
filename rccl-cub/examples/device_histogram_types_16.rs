include!("support/histogram_scalar_probe.rs");

histogram_scalar_probe!("DeviceHistogram 16-bit scalar matrix"; (i16, i32), (u16, i32));
