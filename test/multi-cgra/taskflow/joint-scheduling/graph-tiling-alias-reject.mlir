// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS
// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--tile-task=task-name=tensor axis=0 factor=2' -o /dev/null \
// RUN:   > %t.err 2>&1
// RUN: FileCheck %s --input-file=%t.err --check-prefix=ERROR

// Static function memrefs can alias unless the frontend supplies an explicit
// contract. Facts and the rewrite pass must agree on the fail-closed result.
module {
  func.func @unknown_alias(%input: memref<8xi32>,
                           %output: memref<8xi32>) -> memref<8xi32> {
    %done = taskflow.task @tensor
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
        %value = memref.load %read[%ii] : memref<8xi32>
        memref.store %value, %write[%ii] : memref<8xi32>
        taskflow.hyperblock.yield
      }) : (index) -> ()
      taskflow.yield done_writes(%write : memref<8xi32>)
    }
    return %done : memref<8xi32>
  }
}

// FACTS-NOT: "kind":"tiling"
// ERROR: tiling requires distinct allocations or amoeba.noalias function arguments
