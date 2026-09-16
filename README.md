# Amoeba

Amoeba is an MLIR project for representing and transforming portable
task-level dataflow programs. It owns the Taskflow dialect, generic Taskflow
transformations, and the integration layer used to connect Taskflow programs
to architecture-specific backends.

[Neura](https://github.com/coredac/neura) is currently the first supported
backend and is consumed as a pinned Git submodule. The joint scheduling and
resource-binding pipeline also pins
[cgra-ii-predictor](https://github.com/guosran/cgra-ii-predictor) as its
external task-shape cost predictor. The project is structured so that additional
spatial dataflow backends can be integrated without moving
architecture-specific concepts into the Taskflow core.

## Architecture

```text
Frontend and loop dialects
          |
          v
  Taskflow dialect             portable task abstraction
          |
          v
  Backend integration          backend-specific conversion and optimization
          |
          v
  Neura backend                current multi-CGRA backend
```

The ownership boundary is:

- **Amoeba core** owns the Taskflow dialect and backend-independent passes.
- **Amoeba backend adapters** translate and optimize Taskflow for a particular
  backend.
- **Backend projects** own their target dialects, architecture models,
  lowerings, mapping, and target-specific transformations.

See the [Backend integration guide](include/Backend/README.md) for the detailed
layout, the Neura integration path, and instructions for adding another
backend.

## Repository layout

```text
amoeba/
|-- include/
|   |-- TaskflowDialect/       # Taskflow dialect and pass interfaces
|   |-- Conversion/            # Backend-independent conversion interfaces
|   `-- Backend/               # Public backend integration interfaces
|-- lib/
|   |-- TaskflowDialect/       # Taskflow implementation and generic passes
|   |-- Conversion/            # Backend-independent conversions
|   `-- Backend/               # Backend adapter implementations
|-- thirdparty/
|   |-- neura/                 # Neura Git submodule
|   `-- cgra-ii-predictor/     # Pinned II model and external adapters
|-- tools/
|   `-- mlir-amoeba-opt/       # Amoeba optimizer driver
`-- test/                      # Core, conversion, end-to-end, and backend tests
```

## Prerequisites

Amoeba requires:

- a C++17 compiler, with `clang` and `clang++` used by the reference build;
- CMake, Ninja, and Make;
- `ccache` and `lld` for the reference LLVM build;
- LLVM/MLIR at commit
  [`6146a88f60492b520a36f8f8f3231e15f3cc6082`](https://github.com/llvm/llvm-project/commit/6146a88f60492b520a36f8f8f3231e15f3cc6082);
  and
- Git submodule support.

Running the ML-ranked joint scheduling driver additionally requires Python
3 and PyTorch as declared by `thirdparty/cgra-ii-predictor/pyproject.toml`.

This is the same LLVM revision used by the current
[Neura build instructions](https://github.com/coredac/neura#build-llvm--neura)
and [Amoeba CI](.github/workflows/main.yml).

## Checkout

Clone Amoeba together with its backend dependency:

```bash
git clone --recurse-submodules git@github.com:coredac/amoeba.git
cd amoeba
```

For an existing checkout, initialize or update the pinned dependencies with:

```bash
git submodule update --init --recursive
git submodule status
```

## Build LLVM and MLIR

Clone LLVM, check out the pinned revision, and create an out-of-tree build:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 6146a88f60492b520a36f8f8f3231e15f3cc6082
mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build . --parallel 2
```

The LLVM and MLIR test suites can be checked with:

```bash
cmake --build . --target check-mlir
cmake --build . --target check-clang
```

## Build Amoeba

From the Amoeba repository root, create a separate build directory and point
CMake to the LLVM/MLIR build above:

```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DMLIR_SOURCE_DIR=/path/to/llvm-project/mlir \
  -DMLIR_BINARY_DIR=/path/to/llvm-project/build

make
```

The Neura submodule is added with `EXCLUDE_FROM_ALL`. CMake therefore builds
the Neura targets required by Amoeba without treating the standalone Neura
tools as part of the default Amoeba build.

## Command-line tool

The build produces `mlir-amoeba-opt`, an `mlir-opt`-style driver that registers
the Taskflow dialect, Amoeba conversions, and all backends enabled in the
build:

```bash
./build/tools/mlir-amoeba-opt/mlir-amoeba-opt --help
```

For example, an affine program can be converted to Taskflow with:

```bash
./build/tools/mlir-amoeba-opt/mlir-amoeba-opt input.mlir \
  --convert-affine-to-taskflow \
  -o output.mlir
```

Neura-specific passes and options are registered by the Neura adapter. The
architecture and latency options are exposed as `--neura-architecture-spec`
and `--neura-latency-spec`.

## Joint scheduling and resource binding

`tools/run-joint-program.py` is the production whole-program entry point. It
enumerates semantically witnessed task graphs reachable by bounded fusion,
fission, and tiling sequences; prepares every graph under one versioned pass
contract; forms the complete task-shape Cartesian product; checks the global
candidate budget before prediction; scores the union with the production
spatial-temporal scheduler; and invokes the native mapper only for the global
top-1.

```bash
./tools/run-joint-program.py task-graph.mlir \
  --architecture test/archspec/architecture_4x4.yaml \
  --amoeba-opt build/tools/mlir-amoeba-opt/mlir-amoeba-opt \
  --output-dir /tmp/amoeba-joint-program \
  --workload-domain static_tensor \
  --predictor-checkpoint /path/to/reviewed/checkpoint.json \
  --witness-command 'python3 verify_semantics.py {source} {candidate}'
```

The input is graph-search-phase Taskflow IR: each compute task has a static
counter chain and one hyperblock. Every rewritten graph must pass structural
verification and the caller-supplied semantic witness. Unsupported graph
shapes fail closed instead of being treated as verified. The exact preparation
pipeline then produces one pre-mapper Neura kernel per compute task;
`taskflow.join` remains a non-compute completion barrier.

For each graph, every task independently receives all legal fixed-orientation
rectangles. A 4 by 4 physical fabric therefore exposes all 16 shapes per task.
The search does not require all selected rectangles to fit simultaneously:
half-open temporal schedules may reuse physical resources. The default
`--max-cgras-per-task 0` means the whole physical grid. Per-graph and global
candidate bounds are fail-before-publish limits, not pruning heuristics.

The ranking objective is predicted spatial-temporal scheduler makespan. Each
task duration is
`startup_cycles + predicted_ii * (trip_count - 1)`; the scorer rounds it up to
whole cycles and uses checked signed 64-bit event time. Ranking ties are broken
by graph-manifest index and then candidate-manifest index. The selected graph,
oriented task footprints, placement, routes, and predicted trace are retained;
after task-local native mapping, the same binder emits a distinct realized
trace and makespan.

With `--communication-mode explicit`, the architecture must define a versioned
`inter_task_network`. Data edges carry exact static payload bits and reserve a
deterministically selected load-aware path. Transfer cost is path latency plus
the ceiling of payload bits divided by bottleneck bandwidth, applied once.
Value/RAW, WAR, WAW, channel, and control dependencies remain distinct;
control-only edges do not invent payloads. `--communication-mode none` retains
precedence without charging transfers.

The pipeline artifacts carry explicit SHA-256 provenance. `candidate_manifest_sha256`
is the hash of the exact candidate JSONL bytes, so a cost catalogue cannot be
reused after candidate order or contents change. `architecture_sha256` is the
hash of the exact architecture YAML bytes; grid dimensions alone do not
identify routing, functional-unit, memory, or latency capabilities. Each
`task_body_sha256` binds predictor costs to the current task IR after
joint-scheduling attributes are removed; the task's trip count is kept as a
separate metadata field because it changes duration rather than the task
computation.
`task_dfg_sha256` identifies the extracted DFG report used by the predictor.
The Python adapter binds that report, the Neura executable, model/checkpoint
weights, and their configuration files to their hashes; the C++ scorer validates
the declarations but does not reopen those producer files. The score header
repeats the candidate, architecture, and exact cost-catalogue hashes so the
driver can verify the complete chain immediately before materializing a
shortlist.

The pinned reference model implements 156 standardized features and the
`156 -> 64 -> 32 -> 1` GELU network with 12,161 parameters. Its safe data-only
checkpoint contract covers all 16 oriented 4 by 4 physical-fabric rectangles.
The checked-in real checkpoint is still provisional: its available evidence is
800 successful rows from 50 mapper-visible DFGs, not the presentation's
historical 1,152 rows and 56 DFGs. It therefore requires
`--allow-provisional-checkpoint`; the deterministic fixture separately requires
`--allow-test-fixture` and makes no quality claim. Mapper failures and timeouts
remain censored rather than numeric labels.

`tools/run-joint-scheduling.py` remains available as a lower-level,
single-graph diagnostic driver for already prepared Taskflow/Neura IR. It is
not the whole-program graph-search entry point.

Current graph-rewrite support deliberately fails closed outside its reviewed
static tensor subset. Tiling requires identity-layout memrefs, an alias proof
for function arguments, and one output memref per task. Dynamic domains,
ambiguous effects or payloads, and unsupported control annotations are
rejected. See [the implementation contract](JOINT_SCHEDULING_CONTRACT.md),
[the review report](JOINT_SCHEDULING_REVIEW.md), and
[the graph-space notes](tools/joint_graph_variant_space.md) for the exact
boundary.

## Tests

After building Amoeba, run the complete lit suite from `amoeba/build`:

```bash
cd test
llvm-lit . -v
```

If `llvm-lit` is not on `PATH`, invoke it with the full path instead, for
example `/path/to/llvm-project/build/bin/llvm-lit . -v`.

The suite covers Taskflow conversions and transformations, end-to-end lowering,
and integration with the Neura backend.
