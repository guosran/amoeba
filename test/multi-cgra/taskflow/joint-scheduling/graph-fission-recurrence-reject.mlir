// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--fission-task=task-name=recurrence left-nodes=0' -o /dev/null \
// RUN:   > %t.err 2>&1
// RUN: FileCheck %s --input-file=%t.err

module {
  func.func @reject_recurrence(%input: memref<8xi32>,
      %output: memref<8xi32>, %seed: i32) -> memref<8xi32> {
    %out = taskflow.task @recurrence
        will_reads(%input : memref<8xi32>)
        will_writes(%output : memref<8xi32>)
        value_inputs(%seed : i32)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%output : memref<8xi32>)]
        : (memref<8xi32>, memref<8xi32>, i32) -> (memref<8xi32>) {
    ^bb0(%read: memref<8xi32>, %write: memref<8xi32>, %initial: i32):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c8 step %c1 : index
      %result = "taskflow.hyperblock"(%i, %initial)
          <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%ii: index, %carried: i32):
        %element = memref.load %read[%ii] : memref<8xi32>
        %sum = arith.addi %element, %carried : i32
        %one = arith.constant 1 : i32
        %next = arith.addi %sum, %one : i32
        memref.store %sum, %write[%ii] : memref<8xi32>
        taskflow.hyperblock.yield iter_args_next(%next : i32) results(%sum : i32)
      }) : (index, i32) -> i32
      taskflow.yield done_writes(%write : memref<8xi32>)
    }
    return %out : memref<8xi32>
  }
}

// CHECK: fission rejects recurrence or a non-canonical counter domain
