// RUN: mlir-amoeba-opt %s --analyze-task-pipeline-interval \
// RUN:   --verify-diagnostics

module {
  // expected-error@+1 {{task schedule exceeds the signed 64-bit cycle range}}
  func.func @cycle_overflow(%seed0: i32, %seed1: i32) -> (i32, i32) {
    %t0 = taskflow.task @Task_0 value_inputs(%seed0 : i32)
        {profile_info = {duration = 9223372036854775802 : i64},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%seed1 : i32)
        {profile_info = {duration = 10 : i64},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t0, %t1 : i32, i32
  }
}
