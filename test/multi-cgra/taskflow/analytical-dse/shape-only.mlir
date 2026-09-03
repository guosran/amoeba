// The first Amoeba analytical-DSE slice freezes all rectangular shape
// combinations before consulting the ML costs.  Two tasks and the 1..2-CGRA
// rectangular family give 3 * 3 = 9 candidates.
//
// RUN: split-file %s %t
// RUN: mlir-amoeba-opt %t/input.mlir \
// RUN:   '--enumerate-analytical-task-candidates=output=%t/candidates.jsonl max-candidates=9 max-cgras-per-task=2' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml
// RUN: FileCheck %s --check-prefix=ENUM --input-file=%t/candidates.jsonl
// RUN: mlir-amoeba-opt %t/input.mlir \
// RUN:   '--score-analytical-task-candidates=candidates=%t/candidates.jsonl cost-file=%t/costs.json output=%t/scores.jsonl top-k=2' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml
// RUN: FileCheck %s --check-prefix=SCORE --input-file=%t/scores.jsonl
// RUN: mlir-amoeba-opt %t/input.mlir \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t/candidates.jsonl candidate-index=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   | FileCheck %s --check-prefix=MATERIALIZE
// RUN: %not mlir-amoeba-opt %t/input.mlir \
// RUN:   '--enumerate-analytical-task-candidates=output=%t/rejected.jsonl max-candidates=8 max-cgras-per-task=2' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LIMIT
// RUN: %not test -e %t/rejected.jsonl
// RUN: sed 's/latency: 1/latency: 2/' %S/../../../archspec/architecture_4x4.yaml > %t/changed-architecture.yaml
// RUN: %not mlir-amoeba-opt %t/input.mlir \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t/candidates.jsonl candidate-index=0' \
// RUN:   --architecture-spec=%t/changed-architecture.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-MISMATCH
// RUN: mlir-amoeba-opt %t/derived-body.mlir \
// RUN:   '--enumerate-analytical-task-candidates=output=%t/derived-before.jsonl max-candidates=1 max-cgras-per-task=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml
// RUN: mlir-amoeba-opt %t/derived-body.mlir \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t/derived-before.jsonl candidate-index=0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml > %t/derived-materialized.mlir
// RUN: mlir-amoeba-opt %t/derived-materialized.mlir \
// RUN:   '--enumerate-analytical-task-candidates=output=%t/derived-after.jsonl max-candidates=1 max-cgras-per-task=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml
// RUN: cmp %t/derived-before.jsonl %t/derived-after.jsonl
// RUN: %not mlir-amoeba-opt %t/input.mlir \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t/oversized-header.jsonl candidate-index=0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MANIFEST-ARCH
// RUN: %not mlir-amoeba-opt %t/input.mlir \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t/overflow-shape.jsonl candidate-index=0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SHAPE-OVERFLOW

// ENUM-DAG: "record_type":"header"
// ENUM-DAG: "search_scope":"shape-only-v1"
// ENUM-DAG: "shape_policy":"rectangles-v1"
// ENUM-DAG: "spec_fingerprint":"sha256:1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"
// ENUM-DAG: "mapper_shape_id":"4x4-90f8d32222"
// ENUM-DAG: "mapper_shape_id":"4x8-c2b4921213"
// ENUM-DAG: "mapper_shape_id":"8x4-1b9dcc3d9b"
// ENUM-DAG: "candidate_count":9
// ENUM-DAG: "candidate_ids_sha256":"{{[0-9a-f]{64}[ ]*}}"

// SCORE-DAG: "score_model":"shape-only-compute-bottleneck-v1"
// SCORE-DAG: "candidate_count":9
// SCORE-DAG: "scored_count":9
// SCORE-DAG: "valid_count":9
// SCORE-DAG: "top_k_requested":2
// SCORE-DAG: "shortlist":[
// SCORE-DAG: "hits":12
// SCORE-DAG: "misses":6
// SCORE-DAG: "entries":6

// MATERIALIZE-LABEL: func.func @two_task_shape_dse
// MATERIALIZE-SAME: analytical_task_candidate_id = "{{[0-9a-f]{64}[ ]*}}"
// MATERIALIZE-SAME: analytical_task_candidate_scope = "shape-only-v1"
// MATERIALIZE: taskflow.task @Task_A
// MATERIALIZE-SAME: cgra_count = 1 : i32
// MATERIALIZE-SAME: cgra_shape = "1x1"
// MATERIALIZE: taskflow.task @Task_B
// MATERIALIZE-SAME: cgra_count = 2 : i32
// MATERIALIZE-SAME: cgra_shape = "1x2"

// LIMIT: complete shape space exceeds max-candidates=8
// LIMIT: refusing to publish a partial candidate manifest
// ARCH-MISMATCH: candidate manifest architecture does not match current Neura architecture
// MANIFEST-ARCH: candidate manifest architecture does not match current Neura architecture
// SHAPE-OVERFLOW: candidate contains a non-rectangular or invalid shape

//--- input.mlir
module {
  func.func @two_task_shape_dse(%seed: i32) -> i32 {
    %a = taskflow.task @Task_A value_inputs(%seed : i32)
        {analytical_body_id = "body-a", trip_count = 4 : i32}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }
    %b = taskflow.task @Task_B value_inputs(%a : i32)
        {analytical_body_id = "body-b", trip_count = 4 : i32}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }
    return %b : i32
  }
}

//--- derived-body.mlir
module {
  func.func @derived_body_id(%seed: i32) -> i32 {
    %result = taskflow.task @Task_Derived value_inputs(%seed : i32)
        {trip_count = 4 : i32} : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }
    return %result : i32
  }
}

//--- costs.json
{
  "schema_version": "amoeba-task-shape-cost-v1",
  "function": "two_task_shape_dse",
  "namespace": "test-model-v1",
  "architecture_fingerprint": "sha256:1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6",
  "entries": [
    {"task": "Task_A", "body_id": "body-a", "mapper_shape_id": "4x4-90f8d32222", "support_status": "supported", "predicted_ii": 4.0, "startup_cycles": 2.0},
    {"task": "Task_A", "body_id": "body-a", "mapper_shape_id": "4x8-c2b4921213", "support_status": "supported", "predicted_ii": 2.0, "startup_cycles": 2.0},
    {"task": "Task_A", "body_id": "body-a", "mapper_shape_id": "8x4-1b9dcc3d9b", "support_status": "supported", "predicted_ii": 3.0, "startup_cycles": 2.0},
    {"task": "Task_B", "body_id": "body-b", "mapper_shape_id": "4x4-90f8d32222", "support_status": "supported", "predicted_ii": 5.0, "startup_cycles": 1.0},
    {"task": "Task_B", "body_id": "body-b", "mapper_shape_id": "4x8-c2b4921213", "support_status": "supported", "predicted_ii": 1.0, "startup_cycles": 1.0},
    {"task": "Task_B", "body_id": "body-b", "mapper_shape_id": "8x4-1b9dcc3d9b", "support_status": "supported", "predicted_ii": 2.0, "startup_cycles": 1.0}
  ]
}

//--- oversized-header.jsonl
{"record_type":"header","schema_version":"amoeba-analytical-task-candidates-v1","search_scope":"shape-only-v1","shape_policy":"rectangles-v1","candidate_identity":"shape-candidate-sha256-v1","function":"two_task_shape_dse","architecture":{"grid_rows":9223372036854775807,"grid_cols":9223372036854775807,"per_cgra_tile_rows":4,"per_cgra_tile_cols":4,"spec_fingerprint":"sha256:1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"},"max_cgras_per_task":9223372036854775807,"tasks":[{"task":"Task_A","body_id":"body-a","trip_count":4},{"task":"Task_B","body_id":"body-b","trip_count":4}]}

//--- overflow-shape.jsonl
{"record_type":"header","schema_version":"amoeba-analytical-task-candidates-v1","search_scope":"shape-only-v1","shape_policy":"rectangles-v1","candidate_identity":"shape-candidate-sha256-v1","function":"two_task_shape_dse","architecture":{"grid_rows":4,"grid_cols":4,"per_cgra_tile_rows":4,"per_cgra_tile_cols":4,"spec_fingerprint":"sha256:1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"},"max_cgras_per_task":2,"tasks":[{"task":"Task_A","body_id":"body-a","trip_count":4},{"task":"Task_B","body_id":"body-b","trip_count":4}]}
{"record_type":"candidate","schema_version":"amoeba-analytical-task-candidates-v1","candidate_id":"invalid","task_shapes":[{"task":"Task_A","body_id":"body-a","trip_count":4,"shape":{"kind":"rect","rows":9223372036854775807,"cols":2,"cgra_count":1,"cgra_shape":"9223372036854775807x2","mapper_tile_rows":4,"mapper_tile_cols":4,"mapper_shape_id":"invalid"}},{"task":"Task_B","body_id":"body-b","trip_count":4,"shape":{"kind":"rect","rows":1,"cols":1,"cgra_count":1,"cgra_shape":"1x1","mapper_tile_rows":4,"mapper_tile_cols":4,"mapper_shape_id":"4x4-90f8d32222"}}]}
