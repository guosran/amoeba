// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--fission-task=task-name=effectful left-nodes=0' -o /dev/null \
// RUN:   > %t.err 2>&1
// RUN: FileCheck %s --input-file=%t.err

module {
  func.func @reject_effect(%input: memref<8xi32>,
                           %output: memref<8xi32>) -> memref<8xi32> {
    %out = taskflow.task @effectful
        will_reads(%input : memref<8xi32>)
        will_writes(%output : memref<8xi32>)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%output : memref<8xi32>)]
        : (memref<8xi32>, memref<8xi32>) -> (memref<8xi32>) {
    ^bb0(%read: memref<8xi32>, %write: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c8 step %c1 : index
      "taskflow.hyperblock"(%i) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%ii: index):
        memref.dealloc %read : memref<8xi32>
        %element = memref.load %read[%ii] : memref<8xi32>
        memref.store %element, %write[%ii] : memref<8xi32>
        taskflow.hyperblock.yield
      }) : (index) -> ()
      taskflow.yield done_writes(%write : memref<8xi32>)
    }
    return %out : memref<8xi32>
  }
}

// CHECK: fission rejects nested regions and unknown effects
