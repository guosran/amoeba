// The scheduler and interval analyzer must use the same whole-task latency.
// On a 1x2 grid, Task_2 reuses Task_1's cell at cycle 100. Scheduling from the
// profile depths alone would instead reuse Task_0's cell at cycle 1.

// RUN: mlir-amoeba-opt %s \
// RUN:   '--orchestrate-tasks-on-accelerators=scheduling-mode=spatial-temporal' \
// RUN:   --analyze-task-pipeline-interval \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @scheduler_uses_whole_latency
// CHECK-SAME: task_pipeline_interval_info = {bottleneck_task = "Task_0",
// CHECK-SAME: critical_path = ["Task_0"]
// CHECK-SAME: pipeline_interval = 991 : i32
// CHECK: taskflow.task @Task_0
// CHECK-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}]
// CHECK: taskflow.task @Task_1
// CHECK-SAME: cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}]
// CHECK: taskflow.task @Task_2
// CHECK-SAME: cgra_positions = [{col = 1 : i32, context_id = 1 : i32, row = 0 : i32}]

module {
  func.func @scheduler_uses_whole_latency(%seed0: i32, %seed1: i32,
                                         %seed2: i32) -> (i32, i32, i32) {
    %t0 = taskflow.task @Task_0 value_inputs(%seed0 : i32)
        {cgra_count = 1 : i32, compiled_ii = 10 : i32,
         profile_info = {duration = 1 : i32}, trip_count = 100 : i32}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%seed1 : i32)
        {cgra_count = 1 : i32, compiled_ii = 1 : i32,
         profile_info = {duration = 100 : i32}, trip_count = 1 : i32}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t2 = taskflow.task @Task_2 value_inputs(%seed2 : i32)
        {cgra_count = 1 : i32, compiled_ii = 1 : i32,
         profile_info = {duration = 1 : i32}, trip_count = 1 : i32}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t0, %t1, %t2 : i32, i32, i32
  }
}
