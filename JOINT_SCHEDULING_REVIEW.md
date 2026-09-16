# Joint Scheduling and Resource Binding Review

This checkout was reviewed against
`/home/x/shiran/project/amoeba/docs/presentations/amoeba_joint_scheduling_resource_binding_15slides.tex`.
The public vocabulary is joint scheduling and resource binding. Retired search
terminology is absent from project-owned source, tests, tools, targets, and
documentation.

## Review result

The production path now implements the presentation's search and replay
contract for its declared static tensor subset:

- bounded breadth-first fusion, fission, and tiling sequences, with legal
  actions recomputed after every rewrite and structurally equal graphs
  deduplicated;
- direct producer-consumer and sibling fusion legality checks;
- predecessor-closed fission cuts with recurrence, memory-order, effect, and
  live-value checks;
- static tensor tiling across axes and factors, including remainder coverage,
  cross-tile completion joins, and repeated fission/tiling sequences;
- all oriented rectangular shapes per task, including all 16 on a 4 by 4
  physical grid, without simultaneous-fit pruning;
- typed value/RAW, WAR, WAW, channel, and control dependencies;
- exact static payloads, deterministic load-aware paths, shared-link
  reservations, and checked transfer-cycle arithmetic;
- signed 64-bit event scheduling, half-open task intervals, resource reuse,
  fixed-orientation placement, and whole-program makespan scoring;
- the 156-feature, `156 -> 64 -> 32 -> 1` GELU predictor with 12,161 parameters,
  safe data-only checkpoints, all 16 oriented shapes, and censored failures;
- provenance-bound graph, candidate, cost, score, predicted-trace, and
  realized-trace artifacts;
- deterministic global ranking across every graph and shape tuple, followed by
  exactly one task-local native mapper and binder replay for the global top-1;
- transactional publication: graph, candidate, global-budget, prediction, or
  replay failure publishes no result directory.

The production entry point is `tools/run-joint-program.py`. The older
`tools/run-joint-scheduling.py` is retained only as a single-graph diagnostic
pipeline.

## Fail-closed support boundary

The implementation does not silently generalize beyond the reviewed domain.

- Graph search accepts one static Taskflow counter chain and one hyperblock per
  compute task. Another compiler phase is rejected.
- Tiling currently requires static identity-layout memrefs and one output
  memref. Function arguments require an explicit no-alias proof. Multi-output,
  dynamic, or ambiguous cases are rejected.
- Fission materializes static counter-indexed cut tensors and rejects unsafe
  recurrences, effects, ordering, or unsupported live values.
- Explicit communication requires a versioned `inter_task_network` in the
  architecture. Dynamic or ambiguous payloads are rejected.
- Rewritten graphs require an external semantic witness. The framework records
  its command template and hashes, but the workload-specific equivalence oracle
  is intentionally not fabricated inside the compiler.

These restrictions reduce the accepted input set; they do not weaken the
invariants for accepted candidates.

## Evidence status

The reference model architecture and serving contract match the presentation,
but model-quality promotion is not claimed. The available real checkpoint has
800 successful rows from 50 distinct mapper-visible DFGs. The presentation's
historical 1,152-row, 56-DFG corpus is not present with reproducible manifests
and weights, so the real checkpoint remains provisional and is rejected unless
explicitly enabled. The zero-quality checkpoint is an adapter fixture only.

The repository also does not yet contain the presentation's complete Attention
and SimpleResNet predicted-versus-oracle study. Consequently, task-level held-
out error, shape-ranking quality, whole-program makespan regret, and mapper-time
reduction remain experimental evidence gaps rather than implementation claims.

## Verification performed

- The merged C++/MLIR targets build successfully in a fresh review build.
- The joint-scheduling lit directory passes 19 of 19 tests.
- The runtime contract executable passes, including typed edges, joins,
  transfer accounting, link reservations, temporal reuse, and overflow checks.
- The real task-local mapping bridge test passes.
- Graph-space and global-ranking Python tests pass 7 of 7.
- Predictor and adapter tests pass 35 of 35.
- A real two-task whole-program smoke run scores the full 16 by 16 shape
  product, then performs one native mapper replay. It reports predicted
  makespan 73 and realized makespan 45 with distinct trace hashes.
- A real one-rewrite search carries all 13 discovered graph variants through
  preparation, fixture inference, global ranking, and one mapper replay; a
  transformed graph wins the deterministic global ranking.
- An explicit-communication smoke run derives a 640-bit RAW transfer, reserves
  the local channel for 20 cycles at 32 bits per cycle, and reports predicted
  makespan 93 and realized makespan 65.
- Python syntax, `git diff --check`, C++ formatting, and the retired-term scan
  pass after generated caches are removed.

The smoke run validates integration and replay, not prediction quality. The
provisional evidence gate above remains the honest blocker to an empirical
quality claim.
