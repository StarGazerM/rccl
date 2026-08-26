//! Typed, private compiler data for the Thrust surface.
//!
//! This is not user syntax. Public algorithm macros parse shallow Rust calls
//! into operation-specific `syn` ASTs, then consult this closed catalog for
//! semantic lowering to CUB.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Planned,
    NativeRust,
    Implemented,
    BackendBlocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RustSemantic {
    ComplexValue,
    Tuple,
    Closure,
    ProductClosure,
    RandomState,
    Borrow,
    IndexArithmetic,
    RangeLength,
    AffineOwner,
    Drop,
    DeviceBuffer,
    MaybeUninit,
    Ownership,
    DeviceAllocator,
    ManagedAllocator,
    AllocatorValue,
    DeviceBorrow,
    IndexedAccess,
    ManagedBorrow,
    Vec,
    ManagedBuffer,
    ExecutionBorrow,
    ResultValue,
    CrateVersion,
    RandomDistribution,
    PoolAllocator,
    ThreadLocalAllocator,
    AllocatorAdaptor,
    HostAllocator,
    ConstructionAllocator,
    GenericAllocator,
    PoolConfiguration,
    SynchronizedAllocator,
    CheckedAllocator,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Support {
    pub(crate) cccl: &'static str,
    pub(crate) rust: RustSemantic,
    pub(crate) status: Status,
}

use RustSemantic as RS;
use Status::*;

pub(crate) const SUPPORT: &[Support] = &[
    Support {
        cccl: "complex",
        rust: RS::ComplexValue,
        status: Planned,
    },
    Support {
        cccl: "pair",
        rust: RS::Tuple,
        status: NativeRust,
    },
    Support {
        cccl: "tuple",
        rust: RS::Tuple,
        status: NativeRust,
    },
    Support {
        cccl: "functional",
        rust: RS::Closure,
        status: NativeRust,
    },
    Support {
        cccl: "zip_function",
        rust: RS::ProductClosure,
        status: Planned,
    },
    Support {
        cccl: "random",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "addressof",
        rust: RS::Borrow,
        status: NativeRust,
    },
    Support {
        cccl: "advance",
        rust: RS::IndexArithmetic,
        status: NativeRust,
    },
    Support {
        cccl: "distance",
        rust: RS::RangeLength,
        status: NativeRust,
    },
    Support {
        cccl: "allocate_unique",
        rust: RS::AffineOwner,
        status: Planned,
    },
    Support {
        cccl: "device_delete",
        rust: RS::Drop,
        status: NativeRust,
    },
    Support {
        cccl: "device_free",
        rust: RS::Drop,
        status: NativeRust,
    },
    Support {
        cccl: "device_make_unique",
        rust: RS::AffineOwner,
        status: Planned,
    },
    Support {
        cccl: "device_malloc",
        rust: RS::DeviceBuffer,
        status: Implemented,
    },
    Support {
        cccl: "device_new",
        rust: RS::MaybeUninit,
        status: Planned,
    },
    Support {
        cccl: "memory",
        rust: RS::Ownership,
        status: NativeRust,
    },
    Support {
        cccl: "device_allocator",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "device_malloc_allocator",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "device_new_allocator",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "universal_allocator",
        rust: RS::ManagedAllocator,
        status: Implemented,
    },
    Support {
        cccl: "per_device_resource",
        rust: RS::AllocatorValue,
        status: Implemented,
    },
    Support {
        cccl: "device_ptr",
        rust: RS::DeviceBorrow,
        status: Planned,
    },
    Support {
        cccl: "device_reference",
        rust: RS::IndexedAccess,
        status: Planned,
    },
    Support {
        cccl: "universal_ptr",
        rust: RS::ManagedBorrow,
        status: Planned,
    },
    Support {
        cccl: "device_vector",
        rust: RS::DeviceBuffer,
        status: Implemented,
    },
    Support {
        cccl: "host_vector",
        rust: RS::Vec,
        status: NativeRust,
    },
    Support {
        cccl: "universal_vector",
        rust: RS::ManagedBuffer,
        status: Planned,
    },
    Support {
        cccl: "execution_policy",
        rust: RS::ExecutionBorrow,
        status: Implemented,
    },
    Support {
        cccl: "system_error",
        rust: RS::ResultValue,
        status: NativeRust,
    },
    Support {
        cccl: "version",
        rust: RS::CrateVersion,
        status: NativeRust,
    },
    Support {
        cccl: "discard_block_engine",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "linear_congruential_engine",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "linear_feedback_shift_engine",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "normal_distribution",
        rust: RS::RandomDistribution,
        status: Planned,
    },
    Support {
        cccl: "subtract_with_carry_engine",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "uniform_int_distribution",
        rust: RS::RandomDistribution,
        status: Planned,
    },
    Support {
        cccl: "uniform_real_distribution",
        rust: RS::RandomDistribution,
        status: Planned,
    },
    Support {
        cccl: "xor_combine_engine",
        rust: RS::RandomState,
        status: Planned,
    },
    Support {
        cccl: "mr_allocator",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "device_memory_resource",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "disjoint_pool",
        rust: RS::PoolAllocator,
        status: Implemented,
    },
    Support {
        cccl: "disjoint_sync_pool",
        rust: RS::PoolAllocator,
        status: Implemented,
    },
    Support {
        cccl: "disjoint_tls_pool",
        rust: RS::ThreadLocalAllocator,
        status: Planned,
    },
    Support {
        cccl: "fancy_pointer_resource",
        rust: RS::AllocatorAdaptor,
        status: Planned,
    },
    Support {
        cccl: "host_memory_resource",
        rust: RS::HostAllocator,
        status: Implemented,
    },
    Support {
        cccl: "memory_resource",
        rust: RS::DeviceAllocator,
        status: Implemented,
    },
    Support {
        cccl: "new_resource",
        rust: RS::ConstructionAllocator,
        status: Planned,
    },
    Support {
        cccl: "polymorphic_adaptor",
        rust: RS::GenericAllocator,
        status: NativeRust,
    },
    Support {
        cccl: "pool",
        rust: RS::PoolAllocator,
        status: Implemented,
    },
    Support {
        cccl: "pool_options",
        rust: RS::PoolConfiguration,
        status: Implemented,
    },
    Support {
        cccl: "sync_pool",
        rust: RS::SynchronizedAllocator,
        status: Implemented,
    },
    Support {
        cccl: "tls_pool",
        rust: RS::ThreadLocalAllocator,
        status: Planned,
    },
    Support {
        cccl: "universal_memory_resource",
        rust: RS::ManagedAllocator,
        status: Implemented,
    },
    Support {
        cccl: "validator",
        rust: RS::CheckedAllocator,
        status: Implemented,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RangeCapability {
    Input,
    Output,
    RandomAccess,
    Permutation,
    Strided,
    Product,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Adaptor {
    pub(crate) name: &'static str,
    pub(crate) status: Status,
    pub(crate) capabilities: &'static [RangeCapability],
}

use RangeCapability as RC;

pub(crate) const ADAPTORS: &[Adaptor] = &[
    Adaptor {
        name: "constant_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "counting_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "discard_iterator",
        status: Implemented,
        capabilities: &[RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "iterator_adaptor",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "iterator_categories",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "iterator_facade",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "iterator_traits",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "iterator_traversal_tags",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "offset_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "permutation_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::RandomAccess, RC::Permutation],
    },
    Adaptor {
        name: "retag",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "reverse_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "shuffle_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::RandomAccess, RC::Permutation],
    },
    Adaptor {
        name: "strided_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess, RC::Strided],
    },
    Adaptor {
        name: "tabulate_output_iterator",
        status: Implemented,
        capabilities: &[RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_input_output_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_output_iterator",
        status: Implemented,
        capabilities: &[RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "zip_iterator",
        status: Implemented,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess, RC::Product],
    },
];

pub(crate) const CUB_PRIMITIVES: &[&str] = &[
    "device_adjacent_difference",
    "device_copy",
    "device_find",
    "device_for",
    "device_merge",
    "device_merge_sort",
    "device_partition",
    "device_radix_sort",
    "device_reduce",
    "device_scan",
    "device_select",
    "device_transform",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    InputOutput,
    Query,
    Reduction,
    Output,
    InPlace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Predicate {
    Otherwise,
    NaturalRadixKey,
    Fused,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Selection {
    pub(crate) when: Predicate,
    pub(crate) primitive: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Algorithm {
    pub(crate) name: &'static str,
    pub(crate) status: Status,
    pub(crate) form: Form,
    pub(crate) requires: &'static [&'static str],
    pub(crate) selections: &'static [Selection],
}

const fn select(when: Predicate, primitive: &'static str) -> Selection {
    Selection { when, primitive }
}

macro_rules! algorithm {
    ($name:literal, $status:ident, $form:ident, [$($requires:literal),+], [$($when:ident => $target:literal),+]) => {
        Algorithm { name: $name, status: Status::$status, form: Form::$form, requires: &[$($requires),+], selections: &[$(select(Predicate::$when, $target)),+] }
    };
}

pub(crate) const ALGORITHMS: &[Algorithm] = &[
    algorithm!("adjacent_difference", Planned, InputOutput, ["device_adjacent_difference"], [Otherwise => "device_adjacent_difference"]),
    algorithm!("binary_search", Planned, Query, ["device_find"], [Otherwise => "device_find"]),
    algorithm!("copy", Planned, InputOutput, ["device_copy"], [Otherwise => "device_copy"]),
    algorithm!("count", Planned, Reduction, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("equal", Planned, Query, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("extrema", Planned, Reduction, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("fill", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("find", Planned, Query, ["device_find"], [Otherwise => "device_find"]),
    algorithm!("for_each", Planned, InputOutput, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("gather", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("generate", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("inner_product", Planned, Reduction, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("logical", Planned, Query, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("merge", BackendBlocked, InputOutput, ["device_merge"], [Otherwise => "device_merge"]),
    algorithm!("mismatch", Planned, Query, ["device_find"], [Otherwise => "device_find"]),
    algorithm!("partition", Planned, InPlace, ["device_partition"], [Otherwise => "device_partition"]),
    algorithm!("reduce", Planned, Reduction, ["device_reduce"], [Otherwise => "device_reduce"]),
    algorithm!("remove", Planned, InPlace, ["device_select"], [Otherwise => "device_select"]),
    algorithm!("replace", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("reverse", Planned, InPlace, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("scan", Planned, InputOutput, ["device_scan"], [Otherwise => "device_scan"]),
    algorithm!("scatter", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("sequence", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("set_operations", Planned, InputOutput, ["device_merge"], [Otherwise => "device_merge"]),
    algorithm!("shuffle", Planned, InPlace, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("sort", BackendBlocked, InPlace, ["device_radix_sort", "device_merge_sort"], [NaturalRadixKey => "device_radix_sort", Otherwise => "device_merge_sort"]),
    algorithm!("swap", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("tabulate", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("transform", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("transform_reduce", Planned, Reduction, ["device_transform", "device_reduce"], [Fused => "device_reduce", Otherwise => "device_transform"]),
    algorithm!("transform_scan", Planned, InputOutput, ["device_transform", "device_scan"], [Fused => "device_scan", Otherwise => "device_transform"]),
    algorithm!("uninitialized_copy", Planned, InputOutput, ["device_copy"], [Otherwise => "device_copy"]),
    algorithm!("uninitialized_fill", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("unique", Planned, InPlace, ["device_select"], [Otherwise => "device_select"]),
];

pub(crate) fn validate() -> Result<(), String> {
    unique("support mapping", SUPPORT.iter().map(|item| item.cccl))?;
    unique("adaptor", ADAPTORS.iter().map(|item| item.name))?;
    unique("algorithm", ALGORITHMS.iter().map(|item| item.name))?;
    let cub = CUB_PRIMITIVES.iter().copied().collect::<BTreeSet<_>>();

    for support in SUPPORT {
        let _closed_semantics = (support.rust, support.status);
    }
    for adaptor in ADAPTORS {
        if adaptor.capabilities.is_empty() {
            return Err(format!(
                "adaptor `{}` has no range capability",
                adaptor.name
            ));
        }
        unique(
            "range capability",
            adaptor
                .capabilities
                .iter()
                .map(|capability| match capability {
                    RangeCapability::Input => "input",
                    RangeCapability::Output => "output",
                    RangeCapability::RandomAccess => "random_access",
                    RangeCapability::Permutation => "permutation",
                    RangeCapability::Strided => "strided",
                    RangeCapability::Product => "product",
                }),
        )?;
        let _closed_status = adaptor.status;
    }
    for algorithm in ALGORITHMS {
        if algorithm.requires.is_empty() || algorithm.selections.is_empty() {
            return Err(format!(
                "algorithm `{}` has no CUB lowering",
                algorithm.name
            ));
        }
        let requires = unique("CUB dependency", algorithm.requires.iter().copied())?;
        for dependency in &requires {
            if !cub.contains(dependency) {
                return Err(format!(
                    "algorithm `{}` imports unknown CUB `{dependency}`",
                    algorithm.name
                ));
            }
        }
        let predicates = unique(
            "selection predicate",
            algorithm
                .selections
                .iter()
                .map(|selection| match selection.when {
                    Predicate::Otherwise => "otherwise",
                    Predicate::NaturalRadixKey => "natural_radix_key",
                    Predicate::Fused => "fused",
                }),
        )?;
        if !predicates.contains("otherwise") {
            return Err(format!(
                "algorithm `{}` has no otherwise selection",
                algorithm.name
            ));
        }
        for selection in algorithm.selections {
            if !requires.contains(selection.primitive) {
                return Err(format!(
                    "algorithm `{}` selects undeclared CUB `{}`",
                    algorithm.name, selection.primitive
                ));
            }
        }
        let _closed_semantics = (algorithm.status, algorithm.form);
    }
    Ok(())
}

fn unique<'a>(
    kind: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeSet<&'a str>, String> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name) {
            return Err(format!("duplicate {kind} `{name}`"));
        }
    }
    Ok(unique)
}
