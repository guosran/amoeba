// RUN: mlir-amoeba-opt %s --analyze-task-pipeline-interval \
// RUN:   --verify-diagnostics

module {
  func.func @duration_formula_overflow(%seed: i32) -> i32 {
    // expected-error@+1 {{task Task_0 execution duration exceeds the signed 64-bit cycle range}}
    %t0 = taskflow.task @Task_0 value_inputs(%seed : i32)
        {compiled_ii = 9223372036854775807 : i64,
         profile_info = {duration = 1 : i32}, trip_count = 2 : i32,
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t0 : i32
  }
}
