// Exact scheduling must start an independent task at the first resource
// release event, not at a multiple of that task's own duration. On a 1x1
// machine, [0,1000003) followed by a two-cycle task has makespan 1000005.
// The large first duration also proves that the scheduler does not quantize
// its time axis around the old one-million-cycle threshold.
// RUN: mlir-amoeba-opt %s \
// RUN:   '--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-based-task-orchestration scheduling-mode=spatial-temporal' \
// RUN:   --neura-architecture-spec=%S/../../../archspec/architecture.yaml \
// RUN:   | FileCheck %s
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-based-task-orchestration scheduling-mode=spatial-temproal' \
// RUN:   --neura-architecture-spec=%S/../../../archspec/architecture.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-MODE

module {
  func.func @main(%seed: i32) attributes {
      joint_scheduling_candidate_id = "candidate-test"} {
    %a = taskflow.task @A value_inputs(%seed : i32)
        {amoeba.joint_shape_orientation_fixed, cgra_count = 1 : i32,
         cgra_shape = "1x1", est_latency = 1000003 : i64,
         trip_count = 1 : i64} : (i32) -> i32 {
      ^bb0(%value: i32):
        taskflow.yield values(%value : i32)
    }
    %b = taskflow.task @B value_inputs(%seed : i32)
        {amoeba.joint_shape_orientation_fixed, cgra_count = 1 : i32,
         cgra_shape = "1x1", est_latency = 2 : i64,
         trip_count = 1 : i64} : (i32) -> i32 {
      ^bb0(%value: i32):
        taskflow.yield values(%value : i32)
    }
    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: joint_scheduling_predicted_makespan = 1000005 : i64
// CHECK: taskflow.task @A
// CHECK-SAME: context_id = 0 : i32
// CHECK: taskflow.task @B
// CHECK-SAME: context_id = 1 : i32
// BAD-MODE: unknown task scheduling mode: spatial-temproal
