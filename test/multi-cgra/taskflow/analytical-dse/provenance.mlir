// On a 4x4 grid with at most four CGRAs per task, each task has eight oriented
// rectangular shapes. Of the 64 ordered pairs, exactly two cannot coexist:
// (A=1x4, B=4x1) and (A=4x1, B=1x4). Each pair has area 8 <= 16, so excluding
// them proves that enumeration performs exact fixed-orientation packing rather
// than only checking total area. Temporal reuse must not admit either pair.

// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl max-cgras-per-task=4 max-candidates=62' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND
// RUN: not mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.too-many.jsonl max-cgras-per-task=4 max-candidates=61' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=LIMIT
// RUN: mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.candidates.jsonl candidate-id=candidate-45' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   | FileCheck %s --check-prefix=MATERIALIZED
// RUN: not mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.candidates.jsonl candidate-id=candidate-45' \
// RUN:   --resource-aware-task-optimization \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-REALLOCATE
// RUN: sed -e '/"record_type":"candidate"/d' \
// RUN:   -e 's/"candidate_count":62/"candidate_count":0/' \
// RUN:   %t.candidates.jsonl > %t.empty.jsonl
// RUN: not mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.empty.jsonl candidate-id=candidate-0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY

// A cost catalog must prove that each DFG came from the current task body. The
// fixture carries unrelated all-zero body hashes, so the scorer rejects it
// before looking up or ranking any candidate.
// RUN: not mlir-amoeba-opt %s \
// RUN:   '--score-analytical-task-candidates=candidates=%t.candidates.jsonl cost-file=%S/stale-cost.json output=%t.scores.jsonl top-k=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=STALE

// CANDIDATES: "spec_sha256":"{{[0-9a-f]{64}}}"
// CANDIDATES: "body_sha256":"{{[0-9a-f]{64}}}"
// CANDIDATES: "spatial_capacity_policy":"all-tasks-simultaneous-exact-pack"
// CANDIDATES: "candidate_id":"candidate-0"
// CANDIDATES-NOT: "cgra_shape":"1x4"{{.*}}"cgra_shape":"4x1"
// CANDIDATES-NOT: "cgra_shape":"4x1"{{.*}}"cgra_shape":"1x4"
// CANDIDATES: "candidate_count":62
// BOUND-COUNT-2: amoeba.source_task_body_sha256 = "{{[0-9a-f]{64}}}"
// LIMIT: complete concurrently packable shape space exceeds max-candidates=61
// MATERIALIZED: taskflow.task @A
// MATERIALIZED-SAME: amoeba.analytical_shape_orientation_fixed
// MATERIALIZED-SAME: cgra_count = 4
// MATERIALIZED-SAME: cgra_shape = "1x4"
// MATERIALIZED: taskflow.task @B
// MATERIALIZED-SAME: amoeba.analytical_shape_orientation_fixed
// MATERIALIZED-SAME: cgra_count = 4
// MATERIALIZED-SAME: cgra_shape = "1x4"
// NO-REALLOCATE: resource-aware-task-optimization cannot run after an analytical task candidate has fixed resource shapes
// EMPTY: invalid candidate manifest footer
// STALE: cost catalogue task provenance does not bind the current task body to its source DFG

module {
  func.func @main(%a: memref<16xf32>, %b: memref<16xf32>) {
    %a_read, %a_write = taskflow.task @A
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        {trip_count = 10 : i64}
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    %b_read, %b_write = taskflow.task @B
        will_reads(%b : memref<16xf32>)
        will_writes(%b : memref<16xf32>)
        [original_read_memrefs(%b : memref<16xf32>),
         original_write_memrefs(%b : memref<16xf32>)]
        {trip_count = 10 : i64}
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }
}
