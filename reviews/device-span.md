# Device span adversarial review

- Ledger row: `device-span`
- Baseline: CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`
- Public example: `examples/span-sort.rs`
- Focused tests: `rccl-memory/src/lib.rs` unit tests and compile-fail doctests
- Full gate: `scripts/test-gpu.sh`

## Independent oracles

The host pointer-arithmetic tests construct a synthetic device allocation at a
known non-null address. They compare a subspan pointer against an independently
computed byte address and check nested ranges and every bounds boundary.

The GPU example starts with zero guards around a non-empty subspan, writes a
descending sequence only inside the subspan, sorts through the ordinary public
facade, and checks two independent properties on the device:

1. every element in the subspan equals its exact ascending index; and
2. every element outside the subspan remains zero.

Only scalar error counts return to the host. The test does not accept mere
sortedness, use a CPU fallback, stage the keys through host memory, or calculate
the expected output through the implementation being tested.

## Seeded fault and observed failure

The review temporarily changed the byte-offset calculation from `offset` to
`offset + 1`. The fully qualified focused command

```text
cargo test -p rccl-memory --lib \
  tests::immutable_subspan_offsets_pointer_and_checks_nested_ranges -- --exact
```

executed one test and failed at the pointer assertion: the faulty address was
`4112`, while the independently expected address was `4108`. The mutation was
then removed, and the same command executed one test and passed.

An initial shortened filter omitted the `tests::` module prefix. Cargo returned
success after running zero tests. That is a masking failure in the review
procedure, not evidence of correctness. The repair is to keep the checked-in
gate unfiltered and to record the executed test count for any future focused
fault-injection command; a successful zero-test run is invalid evidence.

## Masking-risk closure

- The example uses a non-zero span and rejects skipped work through exact output
  checks.
- Descending non-zero input prevents untouched/default output from satisfying
  the expected result.
- Exact iota plus guard checks cover the full output, permutation, and
  out-of-range writes; sortedness alone is not used.
- All checks use the public allocation, span, execution, and sort APIs.
- The Cargo target is registered explicitly and appears in both runtime rounds.
- Compile-fail coverage rejects two simultaneously live overlapping mutable
  subspans.
- Unit coverage includes empty end subspans, nested subspans, out-of-bounds
  offset/length, address overflow, disjoint mutable splits, and invalid splits.
- The example emits and checks scalar GPU verdicts without a full-buffer
  readback or CPU scratch path.
- The complete gate runs host tests, documentation tests, public GPU examples,
  and pinned CCCL references twice, exposing stale or state-dependent results.

## Final result

After restoring the implementation, `cargo test --workspace --no-fail-fast`,
warning-denied all-target compilation, shell validation, the 239-header ledger
check, and both host/GPU/reference rounds of `scripts/test-gpu.sh` passed. The
two GPU rounds each reported `device subspan sort with exact output and guard
validation: PASS`.
