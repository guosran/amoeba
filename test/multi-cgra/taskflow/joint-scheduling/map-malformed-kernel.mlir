// RUN: python3 -c 'from pathlib import Path; Path(r"%t.scores").write_text("{\"record_type\":\"header\",\"schema\":\"amoeba-analytical-task-scores\",\"function\":\"main\"}\n{\"record_type\":\"score\",\"schema\":\"amoeba-analytical-task-scores\",\"candidate_id\":\"candidate-0\",\"valid\":true,\"schedule_trace_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"task_costs\":[{\"task\":\"broken\",\"startup_cycles\":1}]}\n{\"record_type\":\"footer\",\"schema\":\"amoeba-analytical-task-scores\",\"scored_count\":1}\n")'
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--map-joint-scheduling-tasks=scores=%t.scores function=main candidate-id=candidate-0 expected-trace-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa expected-score-sha256=ef2cabfe631fdcfffad39ff949d96d1206816d15e817340fa5ee8f73760a3d37' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @main() attributes {joint_scheduling_candidate_id = "candidate-0"} {
    taskflow.task @broken {
      amoeba.joint_shape_orientation_fixed,
      amoeba.selected_mapper_tile_cols = 4 : i64,
      amoeba.selected_mapper_tile_rows = 4 : i64,
      amoeba.selected_trip_count = 2 : i64,
      cgra_count = 1 : i32,
      cgra_shape = "1x1"
    } : () -> () {
      taskflow.yield
    }
    return
  }
}

// CHECK: per-task mapper replay requires exactly one neura.kernel
