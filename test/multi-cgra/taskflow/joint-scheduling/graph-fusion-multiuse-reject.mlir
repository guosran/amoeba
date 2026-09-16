// RUN: mlir-amoeba-opt %s --verify-each --affine-loop-tree-serialization \
// RUN:   --convert-affine-to-taskflow --construct-hyperblock-from-task \
// RUN:   -o %t.taskflow.mlir
// RUN: mlir-amoeba-opt %t.taskflow.mlir --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS

// Both later tasks consume %c. Eliminating Task_0 into either consumer would
// drop a live RAW result, so no producer-consumer action may be advertised.
module {
  func.func @multiuse(%a: memref<8xf32>)
      -> (memref<8xf32>, memref<8xf32>) {
    %c = memref.alloc() : memref<8xf32>
    %d = memref.alloc() : memref<8xf32>
    %e = memref.alloc() : memref<8xf32>
    affine.for %i = 0 to 8 {
      %av = affine.load %a[%i] : memref<8xf32>
      affine.store %av, %c[%i] : memref<8xf32>
    }
    affine.for %i = 0 to 8 {
      %cv = affine.load %c[%i] : memref<8xf32>
      affine.store %cv, %d[%i] : memref<8xf32>
    }
    affine.for %i = 0 to 8 {
      %cv = affine.load %c[%i] : memref<8xf32>
      affine.store %cv, %e[%i] : memref<8xf32>
    }
    return %d, %e : memref<8xf32>, memref<8xf32>
  }
}

// FACTS-NOT: "mode":"producer_consumer","first_task":"Task_0"
