// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.source-facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.source-facts.json --check-prefix=SOURCE-FACTS
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--fission-task=task-name=wide left-nodes=0:1:2' -o %t.once.mlir
// RUN: mlir-amoeba-opt %t.once.mlir --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.once-facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.once-facts.json --check-prefix=ONCE-FACTS
// RUN: mlir-amoeba-opt %t.once.mlir --verify-each \
// RUN:   '--fission-task=task-name=wide.split.1 left-nodes=0' -o %t.twice.mlir
// RUN: FileCheck %s --input-file=%t.twice.mlir
// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.overflow.json max-tile-factor=4 max-fission-actions-per-task=1' \
// RUN:   -o /dev/null > %t.overflow.err 2>&1
// RUN: FileCheck %s --input-file=%t.overflow.err --check-prefix=OVERFLOW

module {
  func.func @repeat_fission(
      %input: memref<8xi32> {amoeba.noalias},
      %output: memref<8xi32> {amoeba.noalias}) -> memref<8xi32> {
    %done = taskflow.task @wide
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
        %element = memref.load %read[%ii] : memref<8xi32>
        %one = arith.constant 1 : i32
        %sum = arith.addi %element, %one : i32
        %two = arith.constant 2 : i32
        %product = arith.muli %sum, %two : i32
        memref.store %product, %write[%ii] : memref<8xi32>
        taskflow.hyperblock.yield
      }) : (index) -> ()
      taskflow.yield done_writes(%write : memref<8xi32>)
    }
    return %done : memref<8xi32>
  }
}

// CHECK-LABEL: func.func @repeat_fission
// CHECK: memref.alloc() : memref<8xi32>
// CHECK: taskflow.task @wide.split.0
// CHECK-SAME: amoeba.fission.left_nodes = array<i64: 0, 1, 2>
// CHECK: memref.store
// CHECK: taskflow.task @wide.split.1.split.0
// CHECK: taskflow.task @wide.split.1.split.1
// CHECK: return
// CHECK-NOT: taskflow.task @wide will_reads
// SOURCE-FACTS: "kind":"fission","left_nodes":[0,1,2],"task_name":"wide"
// ONCE-FACTS: "kind":"fission","left_nodes":[0],"task_name":"wide.split.1"
// OVERFLOW: legal fission cut-set count exceeds max-fission-actions-per-task
