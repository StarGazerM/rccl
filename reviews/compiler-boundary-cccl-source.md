# Compiler-boundary and pinned-source masking review

Date: 2026-08-26

Scope: removal of support-generation macros, source-derived CUB operation
inventory, and the first operation-specific parsers. This is compiler-input
evidence only; no CUB algorithm is exported and no GPU algorithm coverage is
claimed.

## Masking risks checked

- A support test could pass by treating a host slice as device memory.
  Production `DeviceRange` implementations for `[T]` were removed. Unit tests
  now use test-only shape descriptors, and `scripts/test-gpu.sh` rejects a
  reintroduced host-slice implementation.
- A parser could accept extra operands and silently ignore one. The CUB and
  Thrust parsers require exact arity rather than a minimum arity.
- One generic reduce parser could erase the different output roles of
  `ArgMin`, `ArgMax`, `ArgMinMax`, and `ReduceByKey`. They now have distinct
  nominal ASTs and focused role assertions.
- An RCCL-authored operation list could agree with its own tests while drifting
  from CCCL. The source checker independently extracts 105 methods from the
  pinned programs/API tests and diffs them against the typed compiler catalog.
- Documentation examples could be missed because they are embedded in test
  files rather than executable programs. The checker hashes all 378 exact
  `path:example-begin-id` pairs in addition to the 19 executable programs and
  49 API-test files.

## Temporary fault evidence

1. Changed the CUB parser check from `expressions.len() != expected` to
   `expressions.len() < expected`, allowing an extra operand to be discarded.
   The focused test exited 101 at
   `call::tests::wrong_arity_and_the_old_named_temp_syntax_are_rejected` with
   `an extra output must not be ignored` and printed the incorrectly accepted
   `DeviceReduceArgMinCall`.
2. Changed the typed catalog entry `DeviceReduce::Sum` to
   `DeviceReduce::SumTypo`. `scripts/check-cccl-cub-examples.sh` exited 1 and
   displayed the exact `Sum` versus `SumTypo` diff.

Both faults were removed before the final gate. The parser and typed compiler
catalog were subsequently deleted with the entire RCCL proc-macro crate.
GPU launch/output fault injection remains mandatory for real operations;
source-inventory failures cannot stand in for runtime evidence.
