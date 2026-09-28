// Canonical source for the semantic producer/consumer pipeline.
//
// T = C_initial + A * B
// D = T + E
//
// T is an internal allocation with one producer and one consumer.  The
// textual source is the identity owner for the pipeline recognizer; no
// separately handwritten descriptor is used by the Python tests.
module {
  func.func @gemm_then_add_8x8_i32(
      %A: memref<8x8xi32> {amoeba.noalias},
      %B: memref<8x8xi32> {amoeba.noalias},
      %C_initial: memref<8x8xi32> {amoeba.noalias},
      %E: memref<8x8xi32> {amoeba.noalias},
      %D: memref<8x8xi32> {amoeba.noalias}) -> memref<8x8xi32> {
    %T = memref.alloc() : memref<8x8xi32>
    %produced = taskflow.task @producer
        will_reads(%A, %B, %C_initial : memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>)
        will_writes(%T : memref<8x8xi32>)
        [original_read_memrefs(%A, %B, %C_initial : memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>),
         original_write_memrefs(%T : memref<8x8xi32>)]
        {amoeba.semantic.task_kind = "gemm_producer",
         amoeba.semantic.task_role = "producer",
         amoeba.semantic.k_range = array<i64: 0, 8>,
         amoeba.tiling.output_region_lowers = [array<i64: 0, 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8, 8>]}
        : (memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>)
       -> (memref<8x8xi32>) {
    ^bb0(%readA: memref<8x8xi32>, %readB: memref<8x8xi32>,
         %readC: memref<8x8xi32>, %writeT: memref<8x8xi32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %m = taskflow.counter from %c0 to %c8 step %c1 : index
      %n = taskflow.counter parent(%m : index) from %c0 to %c8 step %c1 : index
      "taskflow.hyperblock"(%m, %n)
          <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%mi: index, %ni: index):
        %c = memref.load %readC[%mi, %ni] : memref<8x8xi32>
        %sum = scf.for %ki = %c0 to %c8 step %c1
            iter_args(%acc = %c) -> (i32) {
          %a = memref.load %readA[%mi, %ki] : memref<8x8xi32>
          %b = memref.load %readB[%ki, %ni] : memref<8x8xi32>
          %product = arith.muli %a, %b : i32
          %updated = arith.addi %acc, %product : i32
          scf.yield %updated : i32
        }
        memref.store %sum, %writeT[%mi, %ni] : memref<8x8xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%writeT : memref<8x8xi32>)
    }
    %consumed = taskflow.task @consumer
        will_reads(%produced, %E : memref<8x8xi32>, memref<8x8xi32>)
        will_writes(%D : memref<8x8xi32>)
        [original_read_memrefs(%T, %E : memref<8x8xi32>, memref<8x8xi32>),
         original_write_memrefs(%D : memref<8x8xi32>)]
        {amoeba.semantic.task_kind = "elementwise_consumer",
         amoeba.semantic.task_role = "consumer",
         amoeba.semantic.k_range = array<i64: 0, 8>,
         amoeba.semantic.incoming_edges = ["producer|producer_consumer|tile_local"],
         amoeba.tiling.input_region_lowers = [array<i64: 0, 0>, array<i64: 0, 0>],
         amoeba.tiling.input_region_uppers = [array<i64: 8, 8>, array<i64: 8, 8>],
         amoeba.tiling.input_region_reasons = ["proven_rectangular", "proven_rectangular"],
         amoeba.tiling.output_region_lowers = [array<i64: 0, 0>],
         amoeba.tiling.output_region_uppers = [array<i64: 8, 8>]}
        : (memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>)
       -> (memref<8x8xi32>) {
    ^bb0(%readT: memref<8x8xi32>, %readE: memref<8x8xi32>,
         %writeD: memref<8x8xi32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %m = taskflow.counter from %c0 to %c8 step %c1 : index
      %n = taskflow.counter parent(%m : index) from %c0 to %c8 step %c1 : index
      "taskflow.hyperblock"(%m, %n)
          <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%mi: index, %ni: index):
        %t = memref.load %readT[%mi, %ni] : memref<8x8xi32>
        %e = memref.load %readE[%mi, %ni] : memref<8x8xi32>
        %sum = arith.addi %t, %e : i32
        memref.store %sum, %writeD[%mi, %ni] : memref<8x8xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%writeD : memref<8x8xi32>)
    }
    return %consumed : memref<8x8xi32>
  }
}
