# Joint Scheduling and Resource Binding Contract

This file is the implementation contract corresponding to
`amoeba_joint_scheduling_resource_binding_15slides.tex`.  A capability is
complete only when the production driver exercises it and the emitted
artifacts contain enough information to validate and replay the decision.

## Candidate identity

A whole-program candidate is the ordered product

```text
graph variant
  x one fixed-orientation rectangle per task
  x spatial placement
  x one route per data-carrying task edge
  x a temporal schedule
```

Its stable identity binds all five components.  A task-shape tuple is not a
whole-program candidate by itself.  A graph rewrite sequence is not one
either.  Candidate and score artifacts must bind the exact source IR,
architecture, predictor, graph variant, task bodies, dependency edges, and
enumeration bounds with content hashes.

## Graph variants

The identity graph is always present.  Legal fusion, fission, and tensor
tiling actions are recomputed after each rewrite; enumeration visits bounded
action sequences rather than a hand-written list of fixtures.  Equal
canonical task graphs are emitted once.

- Producer--consumer fusion requires a direct RAW edge, a single-use result,
  and matching iteration domains.
- Sibling fusion requires a shared input, no dependence between siblings,
  and matching iteration domains.
- Fission cuts a task DFG into two nonempty pieces and rejects cuts that break
  recurrence, memory-order, or side-effect constraints.
- Tiling enumerates every static tensor axis and factors from two through the
  configured bound, covers remainder elements, and preserves cross-tile
  dependencies.

A transformed graph is publishable only after its structural verifier and
semantic witness succeed.  Tool failure, timeout, or an absent witness is a
rejection, not an unverified success.

## Shapes and resource binding

On an `R x C` physical multi-CGRA fabric, each task independently receives
every oriented rectangle `(r, c)` with `1 <= r <= R` and `1 <= c <= C`, subject
only to an explicit per-task limit.  A 4 x 4 fabric therefore has all 16
oriented shapes.  Enumeration does not require all task rectangles to fit at
the same instant; temporal scheduling decides whether physical resources can
be reused legally.

Materialization preserves the selected orientation, placement, and task-local
mapper dimensions.  The native mapper must consume that task-local footprint.
No later heuristic may rotate, resize, or reallocate it.

## Dependencies and communication

The normalized task graph retains value/RAW, WAR, WAW, channel, and control
precedence.  Each data-carrying edge records a statically known payload in
bits or a transferred region from which that payload is derived.  Dynamic or
ambiguous payloads are unsupported until represented explicitly.

For a selected route,

```text
transfer_cycles = path_latency
                + ceil(payload_bits / bottleneck_bandwidth_bits_per_cycle)
```

Arithmetic is checked signed 64-bit arithmetic.  Routes are selected with a
deterministic load-aware cost, and overlapping transfers reserve shared links.
The consumer cannot start before the producer and its transfer complete.
Control-only edges carry precedence without inventing a payload.

## Temporal schedule

Task intervals are half-open `[start, end)`.  Tasks with intersecting physical
footprints cannot overlap.  Candidate start times include dependency-ready
events, communication-complete events, and resource-release events.  The
whole-program objective is the resulting schedule makespan; the maximum
single-task duration is diagnostic only.

## Predictor

The reference task cost model consumes 156 standardized, pre-mapper features
for one route-expanded task DFG and one oriented shape.  Its network is
`156 -> 64 -> 32 -> 1` with GELU hidden activations and exactly 12,161 trainable
parameters.  It predicts a nonnegative residual above the analytical II lower
bound and clamps the result to mapper II 20.  Trip count, placement, and the
whole-program schedule are not model inputs.

Model use is fail closed outside the declared static tensor-operation,
feature, architecture, and shape contracts.  Mapper failures and timeouts are
censored records and never numeric labels.  Checkpoints are data-only or are
loaded with a weights-only mechanism.  Promotion requires source-group-held-
out II error, shape ranking, and program-level makespan regret evidence; a
structurally correct fixture model is not a promoted checkpoint.

## Selection and replay

Every legal candidate is scored under the same predictor, placement,
communication, and temporal contracts.  Ranking is deterministic by
predicted makespan followed by canonical candidate index.  The production
driver invokes the complete native lowering and mapper only for top-1 by
default.

Replay must preserve the selected graph variant, task-local oriented shapes,
and mapper footprints.  Those are binding decisions, so any mismatch fails the
run.  The predicted placement, routes, and temporal schedule are retained as
the selection trace; after the native mapper supplies measured task costs, the
same binder and scheduler produce a separate realized trace.  Different
predicted and realized timing is reported as prediction error, not falsely
reported as replay equality.  Both traces and their hashes are published so
the graph/binding invariants and the makespan delta can be audited.

Every manifest is written transactionally: count or budget failure must leave
no partial published artifact.
