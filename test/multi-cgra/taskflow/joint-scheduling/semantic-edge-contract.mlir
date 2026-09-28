// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.positive.json max-tile-factor=4 function=private_partial' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.positive.json --check-prefix=POSITIVE
// RUN: ! mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.negative.json max-tile-factor=4 function=negative_annotation' \
// RUN:   -o /dev/null > %t.negative.err 2>&1
// RUN: FileCheck %s --input-file=%t.negative.err --check-prefix=NEGATIVE

// A direct done_writes -> will_reads state edge may use a private partial
// allocation as its original memref root.  The two half-open descriptors are
// sufficient to prove the transfer tile-local; the generic dependency kind
// remains raw and the semantic annotation supplies its algorithmic role.
module {
  func.func @private_partial(
      %input: memref<8xi32> {amoeba.noalias},
      %partial: memref<8xi32> {amoeba.noalias},
      %output: memref<8xi32> {amoeba.noalias}) -> memref<8xi32> {
    %produced = taskflow.task @producer
        will_reads(%input : memref<8xi32>)
        will_writes(%partial : memref<8xi32>)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%partial : memref<8xi32>)]
        {amoeba.tiling.output_region_lowers = [array<i64: 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8>]}
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
    %consumed = taskflow.task @consumer
        will_reads(%produced : memref<8xi32>)
        will_writes(%output : memref<8xi32>)
        [original_read_memrefs(%partial : memref<8xi32>),
         original_write_memrefs(%output : memref<8xi32>)]
        {amoeba.semantic.incoming_edges = ["producer|partial_sum|tile_local"],
         amoeba.tiling.input_region_lowers = [array<i64: 0>],
         amoeba.tiling.input_region_uppers = [array<i64: 8>],
         amoeba.tiling.input_region_reasons = ["proven_rectangular"],
         amoeba.tiling.output_region_lowers = [array<i64: 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8>]}
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

  // The annotation names no actual incoming edge.  Extraction must fail
  // closed rather than publishing a semantic edge that is absent from SSA.
  func.func @negative_annotation(
      %input: memref<8xi32> {amoeba.noalias},
      %partial: memref<8xi32> {amoeba.noalias},
      %output: memref<8xi32> {amoeba.noalias}) -> memref<8xi32> {
    %produced = taskflow.task @producer
        will_reads(%input : memref<8xi32>)
        will_writes(%partial : memref<8xi32>)
        [original_read_memrefs(%input : memref<8xi32>),
         original_write_memrefs(%partial : memref<8xi32>)]
        {amoeba.tiling.output_region_lowers = [array<i64: 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8>]}
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
    %consumed = taskflow.task @consumer
        will_reads(%produced : memref<8xi32>)
        will_writes(%output : memref<8xi32>)
        [original_read_memrefs(%partial : memref<8xi32>),
         original_write_memrefs(%output : memref<8xi32>)]
        {amoeba.semantic.incoming_edges = ["missing|partial_sum|tile_local"],
         amoeba.tiling.input_region_lowers = [array<i64: 0>],
         amoeba.tiling.input_region_uppers = [array<i64: 8>],
         amoeba.tiling.input_region_reasons = ["proven_rectangular"],
         amoeba.tiling.output_region_lowers = [array<i64: 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8>]}
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

// POSITIVE: "kind":"raw"
// POSITIVE-SAME: "scope":"tile_local"
// POSITIVE-SAME: "semantic_role":"partial_sum"
// POSITIVE-SAME: "semantic_scope":"tile_local"
// POSITIVE-SAME: "region_lower":[0]
// POSITIVE-SAME: "region_upper":[8]
// NEGATIVE: semantic incoming edge source task does not exist: missing
