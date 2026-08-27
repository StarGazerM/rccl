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
    ImplementedPartial,
}

use Status::*;

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
        status: ImplementedPartial,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "counting_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "discard_iterator",
        status: Planned,
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
        status: Planned,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "permutation_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::RandomAccess, RC::Permutation],
    },
    Adaptor {
        name: "retag",
        status: NativeRust,
        capabilities: &[RC::Input],
    },
    Adaptor {
        name: "reverse_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "shuffle_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::RandomAccess, RC::Permutation],
    },
    Adaptor {
        name: "strided_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess, RC::Strided],
    },
    Adaptor {
        name: "tabulate_output_iterator",
        status: Planned,
        capabilities: &[RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_input_output_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_iterator",
        status: Planned,
        capabilities: &[RC::Input, RC::RandomAccess],
    },
    Adaptor {
        name: "transform_output_iterator",
        status: Planned,
        capabilities: &[RC::Output, RC::RandomAccess],
    },
    Adaptor {
        name: "zip_iterator",
        status: Planned,
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
    algorithm!("merge", ImplementedPartial, InputOutput, ["device_merge"], [Otherwise => "device_merge"]),
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
    algorithm!("sort", ImplementedPartial, InPlace, ["device_radix_sort", "device_merge_sort"], [NaturalRadixKey => "device_radix_sort", Otherwise => "device_merge_sort"]),
    algorithm!("swap", Planned, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("tabulate", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("transform", ImplementedPartial, InputOutput, ["device_transform"], [Otherwise => "device_transform"]),
    algorithm!("transform_reduce", Planned, Reduction, ["device_transform", "device_reduce"], [Fused => "device_reduce", Otherwise => "device_transform"]),
    algorithm!("transform_scan", Planned, InputOutput, ["device_transform", "device_scan"], [Fused => "device_scan", Otherwise => "device_transform"]),
    algorithm!("uninitialized_copy", Planned, InputOutput, ["device_copy"], [Otherwise => "device_copy"]),
    algorithm!("uninitialized_fill", Planned, Output, ["device_for"], [Otherwise => "device_for"]),
    algorithm!("unique", Planned, InPlace, ["device_select"], [Otherwise => "device_select"]),
];

pub(crate) fn validate() -> Result<(), String> {
    unique("adaptor", ADAPTORS.iter().map(|item| item.name))?;
    unique("algorithm", ALGORITHMS.iter().map(|item| item.name))?;
    let cub = CUB_PRIMITIVES.iter().copied().collect::<BTreeSet<_>>();

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
