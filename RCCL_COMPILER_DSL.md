# RCCL shallow compile-time boundary

RCCL uses the smallest Rust compile-time mechanism that can express each
semantic decision:

```text
ordinary function / trait / generic
    when rustc monomorphization already expresses the distinction

macro_rules!
    when lowering is fixed syntax substitution and operand preservation

proc macro
    only when expansion must inspect macro-visible tokens and choose a
    concrete CUB target policy, instruction, static storage extent, or
    kernel family
```

No compile-time decision is moved to runtime. Runtime crates contain no target
schedule, type map, `TypeId`, `Any`, registry, strategy object, module
family, or fallback dispatcher.

## User-facing syntax

The DSL is the shallow Rust invocation itself, not a separate `.dsl` file:

```rust,ignore
rccl_thrust::sort!(::<u32>(&exec, &mut keys))?;

rccl_thrust::sort_by!(::<i64>(
    &exec,
    &mut keys,
    |left: &i64, right: &i64| right.cmp(left),
))?;

rccl_thrust::stable_sort_by_key_with!(::<i64, u32>(
    &exec,
    &mut keys,
    &mut values,
    |left: &i64, right: &i64| left.cmp(right),
))?;

let aggregate = rccl_cub::warp_reduce!(::<i32, 32>(
    thread_value,
    |left, right| left + right,
));
```

Rust expressions, closures, paths, and borrows remain Rust syntax. The macros
do not introduce a C++ iterator-pair facade or comparator object hierarchy.
The `_by_key` suffix changes the input shape (keys plus associated values);
`_by` or `_with` supplies an ordinary Rust ordering closure.

## The three semantic layers

### Ordinary Rust

Rust traits and generics represent open behavior and capabilities:

- `DeviceRange`, `ReadRange`, `WriteRange`, and
  `RandomAccessRange` express range capabilities;
- generic workspace owners express type/lifetime relationships;
- `Fn` bounds express user callables;
- associated types and const generics express normal CRTP/TMP relationships
  that rustc can resolve during monomorphization;
- tuples, arrays, `Option`, `Result`, `MaybeUninit`, borrows, and RAII
  replace their C++ library counterparts directly.

These are not macro inputs merely because the corresponding CCCL code uses a
template.

### Declarative macros

`rccl-thrust/src/macros` owns fixed Thrust-to-CUB lowering:

- program construction;
- transform;
- sort and stable sort;
- sort-by-key forms;
- merge and merge-by-key forms.

The macros preserve evaluation-once bindings, allocate private temporary
storage, and invoke the actual CUB dependency. They make no architecture or
algorithm-policy decision. Natural order is an ordinary closure emitted by
the declarative macro.

`rccl-cub/src/macros` owns the fixed device-wide program binder and the
fixed transform launch wrapper.

A subtle Rust boundary matters here: forwarding a metavariable captured as
`ty` makes that fragment opaque to a nested macro. The implemented primitive
ordering surface therefore captures its supported bare primitive token as an
`ident` and forwards it unchanged to the CUB proc macro. The CUB layer—not
the Thrust wrapper—owns the closed primitive layout decision. Extending this
surface to custom/decomposed keys requires an explicit macro-visible metadata
contract; it must not fall back to runtime type inspection.

### Procedural macros

`rccl-cub-macros` is the only proc-macro crate. Every remaining public entry
has a capability that `macro_rules!` or normal generics cannot provide:

| Entry family | Compile-time decision |
|---|---|
| `#[rccl_cub::program]` | validate and transform an inline module, then inject its scoped binder |
| `warp_reduce!`, `warp_reduce_items!` | validate literal warp/item counts and select scalar shuffle/redux implementation |
| `__device_kernels!` | emit the static cuda-oxide kernel family and exact shared-storage variants |
| DeviceMerge operations/workspaces/counts | read the selected SM target, inspect supported type layout, and materialize literal CUB tile/load/shared-memory policy |
| DeviceMergeSort operations/workspaces/counts | select target/type policy, launch bounds, static shared extent, and pass graph |
| DeviceRadixSort operations/workspaces/counts | select ordered representation, bit width, Policy800 scaling, static unions, kernel entries, and pass graph |

The proc-macro entry functions only parse and dispatch. A fixed public
proc-macro function that merely returns `quote!` is rejected by the
repository gate.

## Architecture boundary

`CUDA_OXIDE_TARGET` is read only by
`rccl-cub-macros/src/target.rs`. Merge and radix expansion request the
already-selected target from that internal module. No public macro accepts an
`sm_XX` token, and no runtime value carries architecture.

This prevents the target parameter from being threaded from the public API
through every intermediate layer. The architecture affects only the CUB
policy point that needs it.

## C++ TMP and CRTP mapping

| C++ form | Rust form |
|---|---|
| template parameter used only as a data/callable type | generic type parameter plus trait bounds |
| CRTP interface specialization | trait implementation and associated types/consts |
| overload selected by ownership/range category | distinct trait capability or borrow shape |
| compile-time integer used by device code | const generic |
| closed architecture/type tuning table | CUB proc-macro policy selection emitting literals |
| template-generated kernel name/storage family | CUB proc-macro static item generation |
| function object | ordinary closure, usually `Copy` on device |
| runtime iterator pair | borrowed range whose length and exclusivity are intrinsic |

A type parameter is not automatically “TMP that requires a macro.” A proc
macro is justified only when rustc cannot provide the required closed token
inspection before type checking or when concrete item generation is required.

## Expansion contract

For a CUB call syntax `C`, macro-visible closed policy facts `T`, and the
cuda-oxide build target `G`, expansion implements:

```text
lower_cub(C, T, G)
  = concrete operation source
  + literal schedule operands
  + exact static storage/kernel entry
```

Runtime values remain runtime values: lengths, bit windows, selected
`DoubleBuffer` side, pass counters, launch counts derived from lengths, and
errors. Runtime branches may select routes that are semantically runtime in
CCCL (for example single-tile versus multi-tile by input length), but they do
not discover type or architecture policy.

The Thrust layer is:

```text
lower_thrust(C)
  = evaluation-once bindings
  + private temporary allocation
  + one corresponding CUB macro invocation
  + synchronization required by the Thrust API
```

It contains no CUB kernel and no duplicate policy table.

## Ownership and safety boundary

Generated algorithms use borrowed ranges, typed workspaces, checked disjoint
regions, closures, and cuda-oxide prepared launches. There is no raw-pointer
view, erased comparator, `Arc` module registry, manual ABI vector, PTX
loader, or RCCL compiler step.

The only algorithm-facing `unsafe` source is the exact cuda-oxide module
`load` provenance bind in the two declarative program constructors. Those
macros generate the module and immediately load that same module; users do not
write or receive an unsafe API. Any additional unsafe expression, loader, or
manual launch ABI is rejected by the boundary gate.

## Temporary storage

Direct device-wide CUB calls receive one operation-specific opaque workspace.
The CUB proc macro computes target/type-dependent layout formulas; the runtime
workspace owns only buffers and borrows.

Thrust calls never expose scratch keys, partitions, radix bins, lookback
state, or counters. Their declarative macros allocate one private temporary
transaction, split it through safe affine typed regions, construct the CUB
workspace, invoke CUB, and synchronize.

## Review rule

A change that makes an existing test pass is not sufficient. Each implemented
slice needs an upstream-corresponding public example, focused unit/negative
tests, result validation, and an adversarial iteration showing that a seeded
implementation or launch error makes the relevant test fail. Performance
thresholds and expected results may not be changed to hide the fault.
