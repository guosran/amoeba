// RUN: sed '/^\/\/ -----/,$d' %s | mlir-amoeba-opt - --lower-joint-taskflow-to-host-scf | FileCheck %s --check-prefix=VALID
// RUN: sed '/^\/\/ -----/,$d' %s | mlir-amoeba-opt - --lower-joint-taskflow-to-host-scf=capture-task-states | FileCheck %s --check-prefix=TRACE
// RUN: sed -n '/^\/\/ -----/,$p' %s > %t.bad.mlir
// RUN: ! mlir-amoeba-opt %t.bad.mlir --lower-joint-taskflow-to-host-scf -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT

module {
  func.func @valid(%input: memref<4xi32> {amoeba.noalias},
                   %output: memref<4xi32> {amoeba.noalias}) -> memref<4xi32> {
    %done = taskflow.task @copy
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> (memref<4xi32>) {
      ^bb0(%read: memref<4xi32>, %write: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %i = taskflow.counter from %c0 to %c4 step %c1 : index
        "taskflow.hyperblock"(%i)
            <{operandSegmentSizes = array<i32: 1, 0>}> ({
        ^bb0(%ii: index):
          %value = memref.load %read[%ii] : memref<4xi32>
          memref.store %value, %write[%ii] : memref<4xi32>
          taskflow.hyperblock.yield
        }) : (index) -> ()
        taskflow.yield done_writes(%write : memref<4xi32>)
    }
    return %done : memref<4xi32>
  }
}

// -----

module {
  func.func @malformed(%input: memref<4xi32> {amoeba.noalias},
                       %output: memref<4xi32> {amoeba.noalias})
      -> memref<4xi32> {
    %done = taskflow.task @bad
        will_reads(%input : memref<4xi32>)
        will_writes(%output : memref<4xi32>)
        [original_read_memrefs(%input : memref<4xi32>),
         original_write_memrefs(%output : memref<4xi32>)]
        : (memref<4xi32>, memref<4xi32>) -> (memref<4xi32>) {
      // The two original_* operands are provenance only.  They must not be
      // counted as body arguments, so this one-argument body is malformed.
      ^bb0(%read: memref<4xi32>):
        taskflow.yield done_writes(%read : memref<4xi32>)
    }
    return %done : memref<4xi32>
  }
}

// VALID-LABEL: func.func @valid
// VALID-NOT: taskflow.
// VALID: scf.for
// VALID: memref.load
// VALID: memref.store
// VALID: return

// TRACE: memref.global @__orbit_trace_valid_0_0 : memref<4xi32> = uninitialized
// TRACE: memref.copy
// TRACE-SAME: amoeba.host.trace_symbol = "__orbit_trace_valid_0_0"

// REJECT: body arguments must match
