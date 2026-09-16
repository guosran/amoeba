// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.facts.json max-tile-factor=3' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.facts.json --check-prefix=FACTS
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--tile-task=task-name=tensor axis=1 factor=2' -o %t.axis1.mlir
// RUN: FileCheck %s --input-file=%t.axis1.mlir --check-prefix=AXIS1
// RUN: mlir-amoeba-opt %t.axis1.mlir --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.axis1-facts.json max-tile-factor=3' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.axis1-facts.json --check-prefix=JOIN-DEPS
// RUN: mlir-amoeba-opt %s --verify-each \
// RUN:   '--fission-task=task-name=tensor left-nodes=0:1' -o %t.fission.mlir
// RUN: mlir-amoeba-opt %t.fission.mlir --verify-each \
// RUN:   '--extract-joint-task-graph-facts=output=%t.fission-facts.json max-tile-factor=3' \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.fission-facts.json --check-prefix=FISSION-FACTS
// RUN: mlir-amoeba-opt %t.fission.mlir --verify-each \
// RUN:   '--tile-task=task-name=tensor.split.1 axis=1 factor=2' \
// RUN:   -o %t.fission-then-tile.mlir
// RUN: FileCheck %s --input-file=%t.fission-then-tile.mlir --check-prefix=FISSION-TILE
// RUN: mlir-amoeba-opt %t.axis1.mlir --verify-each \
// RUN:   '--fission-task=task-name=tensor.tile.1.0 left-nodes=0:1' \
// RUN:   -o %t.tile-then-fission.mlir
// RUN: FileCheck %s --input-file=%t.tile-then-fission.mlir --check-prefix=TILE-FISSION
// RUN: mlir-amoeba-opt %t.axis1.mlir --verify-each \
// RUN:   '--tile-task=task-name=tensor.tile.1.0 axis=0 factor=2' \
// RUN:   -o %t.repeated.mlir
// RUN: FileCheck %s --input-file=%t.repeated.mlir --check-prefix=REPEAT
// RUN: mlir-amoeba-opt %t.repeated.mlir --verify-each \
// RUN:   --classify-task-and-counter --convert-taskflow-to-neura \
// RUN:   --lower-affine --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:   --assign-accelerator \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_with_counter.yaml \
// RUN:   --lower-memref-to-neura --lower-arith-to-neura \
// RUN:   --lower-builtin-to-neura --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const --fold-constant \
// RUN:   --canonicalize-return --canonicalize-live-in \
// RUN:   --leverage-predicated-value --transform-ctrl-to-data-flow \
// RUN:   --fold-constant --insert-data-mov \
// RUN:   -o %t.prepared.mlir
// RUN: FileCheck %s --input-file=%t.prepared.mlir --check-prefix=PREPARED-TASKS
// RUN: FileCheck %s --input-file=%t.prepared.mlir --check-prefix=PREPARED-KERNELS
// RUN: FileCheck %s --input-file=%t.prepared.mlir --check-prefix=PREPARED-JOINS
// RUN: FileCheck %s --input-file=%t.prepared.mlir --check-prefix=PREPARED-MOVEMENT

module {
  func.func @rank_two_axis(
      %input: memref<4x5xi32> {amoeba.noalias},
      %output: memref<4x5xi32> {amoeba.noalias},
      %final: memref<4x5xi32> {amoeba.noalias},
      %bias: i32) -> memref<4x5xi32> {
    %done = taskflow.task @tensor
        will_reads(%input : memref<4x5xi32>)
        will_writes(%output : memref<4x5xi32>)
        value_inputs(%bias : i32)
        [original_read_memrefs(%input : memref<4x5xi32>),
         original_write_memrefs(%output : memref<4x5xi32>)]
        : (memref<4x5xi32>, memref<4x5xi32>, i32)
       -> (memref<4x5xi32>) {
    ^bb0(%read: memref<4x5xi32>, %write: memref<4x5xi32>, %value: i32):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c4 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c5 step %c1 : index
      "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%ii: index, %jj: index):
        %element = memref.load %read[%ii, %jj] : memref<4x5xi32>
        %sum = arith.addi %element, %value : i32
        memref.store %sum, %write[%ii, %jj] : memref<4x5xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%write : memref<4x5xi32>)
    }
    %consumed = taskflow.task @consumer
        will_reads(%done : memref<4x5xi32>)
        will_writes(%final : memref<4x5xi32>)
        [original_read_memrefs(%output : memref<4x5xi32>),
         original_write_memrefs(%final : memref<4x5xi32>)]
        : (memref<4x5xi32>, memref<4x5xi32>) -> (memref<4x5xi32>) {
    ^bb0(%read: memref<4x5xi32>, %write: memref<4x5xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c4 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c5 step %c1 : index
      "taskflow.hyperblock"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> ({
      ^bb0(%ii: index, %jj: index):
        %element = memref.load %read[%ii, %jj] : memref<4x5xi32>
        memref.store %element, %write[%ii, %jj] : memref<4x5xi32>
        taskflow.hyperblock.yield
      }) : (index, index) -> ()
      taskflow.yield done_writes(%write : memref<4x5xi32>)
    }
    return %consumed : memref<4x5xi32>
  }
}

// FACTS: "axis":0,"factor":2,"kind":"tiling","task_name":"tensor"
// FACTS: "axis":1,"factor":3,"kind":"tiling","task_name":"tensor"

// AXIS1-LABEL: func.func @rank_two_axis
// AXIS1: taskflow.task @tensor.tile.1.0
// AXIS1-SAME: amoeba.tiling.derived_range = array<i64: 0, 3>
// AXIS1: taskflow.task @tensor.tile.1.1
// AXIS1-SAME: amoeba.tiling.derived_range = array<i64: 3, 5>
// AXIS1: taskflow.join states({{.*}}) base({{.*}}) axis(1) region([0, 0], [4, 5])

// JOIN-DEPS: "source_task":"tensor.tile.1.0","target_task":"consumer"
// JOIN-DEPS: "source_task":"tensor.tile.1.1","target_task":"consumer"
// JOIN-DEPS: "kind":"fission","left_nodes":[0,1],"task_name":"tensor.tile.1.0"

// FISSION-FACTS: "axis":1,"factor":2,"kind":"tiling","task_name":"tensor.split.1"
// FISSION-TILE: taskflow.task @tensor.split.0
// FISSION-TILE: taskflow.task @tensor.split.1.tile.1.0
// FISSION-TILE: taskflow.task @tensor.split.1.tile.1.1
// FISSION-TILE: taskflow.join

// TILE-FISSION: taskflow.task @tensor.tile.1.0.split.0
// TILE-FISSION: taskflow.task @tensor.tile.1.0.split.1
// TILE-FISSION: taskflow.task @tensor.tile.1.1
// TILE-FISSION: taskflow.join

// REPEAT-LABEL: func.func @rank_two_axis
// REPEAT: taskflow.task @tensor.tile.1.0.tile.0.0
// REPEAT: taskflow.task @tensor.tile.1.0.tile.0.1
// REPEAT: taskflow.join
// REPEAT: taskflow.task @tensor.tile.1.1
// REPEAT: taskflow.join

// PREPARED-TASKS-COUNT-4: taskflow.task
// PREPARED-KERNELS-COUNT-4: neura.kernel
// PREPARED-JOINS-COUNT-2: taskflow.join
// PREPARED-MOVEMENT: neura.data_mov
