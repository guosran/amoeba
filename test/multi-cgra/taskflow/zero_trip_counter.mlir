// RUN: mlir-amoeba-opt %s \
// RUN:   '--resource-aware-task-optimization=balance-skip-mapper=true' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @zero_trip_counter
// CHECK: taskflow.task @Task_0
// CHECK-SAME: trip_count = 0 : i32

module {
  func.func @zero_trip_counter(%seed: i32) -> i32 {
    %t0 = taskflow.task @Task_0 value_inputs(%seed : i32)
        {cgra_count = 4 : i32, compiled_ii = 1 : i32,
         profile_info = {duration = 1 : i32}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c0 step %c1 : index
      taskflow.yield values(%arg0 : i32)
    }

    return %t0 : i32
  }
}
