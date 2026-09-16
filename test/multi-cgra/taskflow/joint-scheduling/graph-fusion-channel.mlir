// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=4' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--fuse-task=first-task=producer second-task=consumer fusion-kind=producer-consumer' \
// RUN:   -o %t.fused.mlir
// RUN: FileCheck %s --input-file=%t.fused.mlir --check-prefix=FUSED

module {
  func.func @channel_raw(%input: memref<8xi32>, %middle: memref<8xi32>,
                         %output: memref<8xi32>) -> memref<8xi32> {
    %produced = taskflow.task @producer
        will_reads(%input : memref<8xi32>)
        will_writes(%middle : memref<8xi32>)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%middle : memref<8xi32>)]
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
    %carried = taskflow.channel %produced : memref<8xi32> -> memref<8xi32>
    %consumed = taskflow.task @consumer
        will_reads(%carried : memref<8xi32>)
        will_writes(%output : memref<8xi32>)
        [original_read_memrefs(%middle : memref<8xi32>),
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
    return %consumed : memref<8xi32>
  }
}

// FACTS: "first_task":"producer","kind":"fusion","mode":"producer_consumer","second_task":"consumer"
// FUSED: taskflow.task @producer.fuse.consumer
// FUSED-NOT: taskflow.channel
// FUSED-NOT: taskflow.task @producer will_reads
// FUSED-NOT: taskflow.task @consumer will_reads
