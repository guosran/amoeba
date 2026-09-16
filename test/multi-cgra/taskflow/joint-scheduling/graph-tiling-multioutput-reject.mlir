// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS
// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--tile-task=task-name=two_outputs axis=0 factor=2' -o /dev/null \
// RUN:   > %t.err 2>&1
// RUN: FileCheck %s --input-file=%t.err --check-prefix=ERROR

module {
  func.func @reject_multioutput(
      %input: memref<8xi32> {amoeba.noalias},
      %first: memref<8xi32> {amoeba.noalias},
      %second: memref<8xi32> {amoeba.noalias})
      -> (memref<8xi32>, memref<8xi32>) {
    %first_done, %second_done = taskflow.task @two_outputs
        will_reads(%input : memref<8xi32>)
        will_writes(%first, %second : memref<8xi32>, memref<8xi32>)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%first, %second : memref<8xi32>, memref<8xi32>)]
        : (memref<8xi32>, memref<8xi32>, memref<8xi32>)
       -> (memref<8xi32>, memref<8xi32>) {
    ^bb0(%read: memref<8xi32>, %write0: memref<8xi32>, %write1: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c8 step %c1 : index
      "taskflow.hyperblock"(%i) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%ii: index):
        %value = memref.load %read[%ii] : memref<8xi32>
        memref.store %value, %write0[%ii] : memref<8xi32>
        memref.store %value, %write1[%ii] : memref<8xi32>
        taskflow.hyperblock.yield
      }) : (index) -> ()
      taskflow.yield done_writes(%write0, %write1 : memref<8xi32>, memref<8xi32>)
    }
    return %first_done, %second_done : memref<8xi32>, memref<8xi32>
  }
}

// FACTS-NOT: "kind":"tiling"
// ERROR: tiling requires one output state
