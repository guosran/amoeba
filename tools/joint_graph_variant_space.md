# Joint graph-variant enumeration

`joint_graph_variant_space.py` is the single graph-space entry point used
before shape, placement, and scheduling enumeration. It performs a bounded
breadth-first traversal. Every child IR is verified, graph facts and legal
actions are recomputed from that child, and a semantic witness is run against
the original program. A witness failure, timeout, variant-budget overflow, or
publication error leaves both requested outputs absent.

Example:

```sh
python3 tools/joint_graph_variant_space.py \
  --input workload.mlir \
  --amoeba-opt build/tools/mlir-amoeba-opt/mlir-amoeba-opt \
  --output graph-variants.jsonl \
  --artifact-dir graph-variants \
  --function main \
  --max-depth 3 \
  --max-variants 1000 \
  --max-tile-factor 8 \
  --max-fission-actions-per-task 256 \
  --witness-command 'python3 verify_semantics.py {source} {candidate}'
```

The JSONL header binds the exact source and optimizer hashes and all search
bounds. Unique `variant` records contain `graph_variant_id`, parent, complete
action history, result IR path/hash, task facts, dependencies, shape needs,
and verified witness provenance. The identity graph is always the first
variant. Structurally duplicate rewrite sequences are retained only as
`duplicate` provenance records, never as downstream IR inputs. The final
footer marks completeness and gives exact unique, duplicate, and attempted
action counts.

Each `result_ir.path` is relative to the header's `artifact_dir`; staging
paths are never published. `graph_variant_id` and `graph_variant_sha256`
form the strict provenance pair passed to shape enumeration and scoring.
The header also binds preparation contract
`amoeba-graph-variant-preparation-v1`. A canonical graph variant is prepared
for prediction with this exact ordered command (the architecture file is an
explicit input to the accelerator-assignment pass):

```sh
mlir-amoeba-opt graph-variants/graph-000001.mlir --verify-each \
  --classify-task-and-counter --convert-taskflow-to-neura \
  --lower-affine --convert-scf-to-cf --convert-cf-to-llvm \
  --assign-accelerator --architecture-spec=architecture.yaml \
  --lower-memref-to-neura --lower-arith-to-neura \
  --lower-builtin-to-neura --lower-llvm-to-neura \
  --promote-input-arg-to-const --fold-constant \
  --canonicalize-return --canonicalize-live-in \
  --leverage-predicated-value --transform-ctrl-to-data-flow \
  --fold-constant --insert-data-mov -o predictor-ready.mlir
```

`--insert-data-mov` is required exactly once, after the graph preparation
passes and before feature extraction. Each compute task must then contain
exactly one pre-mapper `neura.kernel`; `taskflow.join` is a non-compute
completion barrier and never receives a shape or mapper query.

The facts pass can also be inspected directly:

```sh
mlir-amoeba-opt workload.mlir --verify-each \
  '--extract-joint-task-graph-facts=output=facts.json function=main max-tile-factor=8 max-fission-actions-per-task=256' \
  -o /dev/null
```

The graph-search IR is deliberately one phase: every compute task must have a
static `taskflow.counter` chain and exactly one `taskflow.hyperblock`, before
classification and Taskflow-to-Neura conversion. Inputs in another phase fail
closed.

Tensor tiling currently accepts static, identity-layout, single-output tensor
tasks. It uses `axis` plus a factor `k`: it creates exactly `k`
contiguous, domain-covering tasks and distributes any remainder over the
first partitions. The tiles share the input state and publish disjoint region
states through a variadic `taskflow.join`; this preserves parallelism while a
downstream consumer waits for every tile. Each written memref would need its
own join, so multi-output tasks are rejected by this version. Distinct local
allocations are accepted directly. Function memref arguments require the
explicit `amoeba.noalias` argument attribute; without an alias proof, tiling
is not advertised as a legal action.

Fission actions use stable `left_nodes` arrays of hyperblock node ordinals.
The extractor enumerates every legal predecessor-closed nonempty proper
partition, keeps externally visible stores on the right, and materializes
each live scalar cut edge as a static counter-indexed temporary tensor. If a
task has more legal partitions than `max-fission-actions-per-task`, fact
extraction fails without publishing a partial action list.
