// RUN: mlir-amoeba-opt %s --verify-each --affine-loop-tree-serialization \
// RUN:   --convert-affine-to-taskflow --construct-hyperblock-from-task \
// RUN:   -o %t.taskflow.mlir
// RUN: mlir-amoeba-opt %t.taskflow.mlir --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS
// RUN: mlir-amoeba-opt %t.taskflow.mlir --verify-each \
// RUN:   '--fuse-task=first-task=Task_0 second-task=Task_2 fusion-kind=producer-consumer' \
// RUN:   -o %t.fused.mlir
// RUN: FileCheck %s --input-file=%t.fused.mlir --check-prefix=FUSED

// Task_1 is independent of Task_0's output, so delaying Task_0 to Task_2's
// position is legal. The facts pass must inspect all pairs rather than only
// adjacent tasks. Task_0 and Task_1 are also legal siblings through %a.
module {
  func.func @all_pairs(%a: memref<8xf32>, %b: memref<8xf32>)
      -> (memref<8xf32>, memref<8xf32>) {
    %c = memref.alloc() : memref<8xf32>
    %d = memref.alloc() : memref<8xf32>
    %e = memref.alloc() : memref<8xf32>
    affine.for %i = 0 to 8 {
      %av = affine.load %a[%i] : memref<8xf32>
      %bv = affine.load %b[%i] : memref<8xf32>
      %sum = arith.addf %av, %bv : f32
      affine.store %sum, %c[%i] : memref<8xf32>
    }
    affine.for %i = 0 to 8 {
      %av = affine.load %a[%i] : memref<8xf32>
      %two = arith.constant 2.0 : f32
      %scaled = arith.mulf %av, %two : f32
      affine.store %scaled, %d[%i] : memref<8xf32>
    }
    affine.for %i = 0 to 8 {
      %cv = affine.load %c[%i] : memref<8xf32>
      %one = arith.constant 1.0 : f32
      %shifted = arith.addf %cv, %one : f32
      affine.store %shifted, %e[%i] : memref<8xf32>
    }
    return %d, %e : memref<8xf32>, memref<8xf32>
  }
}

// FACTS: "first_task":"Task_0","kind":"fusion","mode":"sibling","second_task":"Task_1"
// FACTS: "first_task":"Task_0","kind":"fusion","mode":"producer_consumer","second_task":"Task_2"

// FUSED-LABEL: func.func @all_pairs
// FUSED-NOT: taskflow.task @Task_0
// FUSED: taskflow.task @Task_1
// FUSED: taskflow.task @Task_0.fuse.Task_2
// FUSED-NOT: taskflow.task @Task_2
// FUSED: return
