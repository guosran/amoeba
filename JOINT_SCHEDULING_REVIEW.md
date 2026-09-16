# Joint Scheduling and Resource Binding Review

This checkout is reviewed against
`/home/x/shiran/project/amoeba/docs/presentations/amoeba_joint_scheduling_resource_binding_15slides.tex`.
It deliberately starts from the clean spatial-shape plus ML-ranking stack at
commit `46f2c9a0af7b324c536d0c700f2cbeecaee60a69`. The later aggregate delivery
was not used as a source baseline because it mixes incomplete future work with
generated evidence and does not form a standalone Git checkout.

## Scope represented by this checkout

Implemented:

- shared task metadata and static trip-count inference;
- complete fixed-orientation rectangular shapes for the configured grid;
- deterministic, provenance-bound, fail-closed candidate manifests;
- exact candidate materialization with stale derived state removed;
- 64-bit event-based spatial-temporal scheduling and makespan reporting;
- deterministic shortlist ordering and top-entry replay.

Pending review:

- ML task-shape cost prediction and its applicability contract;
- production-quality 4×4 model coverage;
- agreement between predictor, scheduler, and final replay.

Planned and intentionally not claimed as implemented:

- general fusion, fission, and tensor-tiling graph variants;
- communication edges, link binding, and transfer-cycle costs;
- transformed-graph temporal lowering and repeated rewrite search.

## Corrections made in this review

- Removed the retired terminology from project-owned paths, namespaces,
  targets, logs, tests, documentation, and the public driver name.
- Added an explicit manifest budget contract and retained atomic publication.
- Changed scheduling from scaled integer stepping to exact 64-bit ready and
  resource-release events.
- Made task dependencies visible through `taskflow.channel` chains.
- Rejects unknown scheduling modes instead of silently selecting a mode.
- Rejects non-converged fixed-shape placement instead of ranking an arbitrary
  iteration.
- Clears stale placement and profile attributes before materializing a new
  candidate.
- Requires unsupported predictor records to be censored rather than carrying
  numeric timing values.
- Fixed function selection so the driver no longer passes an unsupported
  extractor option; selected task names must be globally unique.

## Blocking gaps before a full presentation-level claim

1. The pinned predictor permits only its recorded 1×1 architecture identity,
   while the physical candidate flow is intended for a 4×4 multi-CGRA grid.
2. Its deployed ensemble is not the presentation's 156→64→32→1 reference MLP.
3. Its protocol covers only eight task shapes (up to four physical CGRAs), not
   all 16 oriented rectangles on a 4×4 grid.
4. The current spatial candidate producer requires simultaneous packing. It
   therefore cannot represent tuples that become legal only through temporal
   reuse; that capability belongs to the planned temporal extension.
5. Model applicability is not yet bound to an explicit tensor-operation
   contract. The current driver must remain marked pending review.
6. The predictor adapter's public `--checkpoint` path uses unsafe pickle-based
   loading. The outer driver does not expose that option, but the adapter must
   move to weights-only loading or a data-only checkpoint format.
7. Communication-aware placement and timing are absent by design in this
   foundation and must not be inferred from scheduler makespan alone.
8. The current mapper takes one module-level array size and cannot honor
   different selected shapes for different Taskflow tasks. The driver now
   fails closed for multi-CGRA mapper replay; a per-task mapping bridge remains
   required.

The smoke test using a single candidate proves the artifact pipeline, not
general 4×4 model quality. A release claim requires a validated checkpoint and
an end-to-end test that enumerates multiple 4×4 candidates, ranks them, replays
the top entry, and compares the predicted schedule with the final mapping.

## Verification performed

- `mlir-amoeba-opt` configured and built successfully in a fresh build tree.
- Nine focused joint-scheduling/orchestration lit tests pass, including the
  new event-time, channel-dependency, stale-state, censored-cost, budget, and
  invalid-mode checks.
- The real-checkpoint FIR single-CGRA replay test passes and reports the mapper
  and binding status explicitly.
- All 25 tests that do not require the optional symbol-generation environment
  or the long-running ResNet mapper pass, plus one unsupported test, complete
  without failure.
- `git diff --check`, LLVM C++ formatting checks, Python syntax checks, and the
  retired-terminology source scan pass.

The five symbol-generated tests require a Python environment with
`torch-mlir`; the default interpreter does not provide it. The standalone
ResNet test remains dominated by the known unbounded mapper-backtracking path
and was stopped rather than reported as passed. These are not counted in the
successful test totals above.
