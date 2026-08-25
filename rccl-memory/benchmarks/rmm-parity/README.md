# Pinned RMM comparison

Build RMM commit `c65202f52766e12c73e293b16d6a520a069d4afc`, then point
CMake at that build tree:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/rmm/build
cmake --build build --config Release
LD_LIBRARY_PATH=/path/to/rmm/build:/path/to/rmm/build/_deps/rapids_logger-build \
  build/rmm_allocation_trace --resource=pool --allocations=10000 \
  --max-allocation=1048576 --max-live-bytes=268435456 \
  --warmups=2 --repetitions=7 --seed=24301
```

The C++ and Rust executables share the same deterministic generator and JSON
schema. Compare only runs made on the same GPU/driver with identical arguments.
