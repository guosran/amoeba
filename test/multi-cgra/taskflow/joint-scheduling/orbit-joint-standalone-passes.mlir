// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   --fuse-producer-consumer | FileCheck %s --check-prefix=FUSE
// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   '--tile-output-m=tile-size=4' '--tile-output-n=tile-size=4' \
// RUN:   --fuse-tile-local-pair | FileCheck %s --check-prefix=TILE-FUSE
// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   --fuse-producer-consumer '--tile-output-m=tile-size=4' \
// RUN:   '--tile-output-n=tile-size=4' | FileCheck %s --check-prefix=FUSE-TILE
// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   '--tile-reduction-k-sequential=tile-size=4' \
// RUN:   --fuse-reduction-consumer | FileCheck %s --check-prefix=SEQ
// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   '--tile-reduction-k-parallel-linear=tile-size=4' \
// RUN:   --materialize-numeric-reduction --fuse-reduction-consumer \
// RUN:   | FileCheck %s --check-prefix=LINEAR
// RUN: mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   '--tile-reduction-k-parallel-tree=tile-size=4' \
// RUN:   --materialize-numeric-reduction --fuse-reduction-consumer \
// RUN:   | FileCheck %s --check-prefix=TREE
// RUN: ! mlir-amoeba-opt %S/orbit_joint_pipeline_8x8_i32.mlir --verify-each \
// RUN:   --fuse-reduction-consumer -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REJECT-EARLY

// FUSE: taskflow.task @producer.fuse.consumer
// FUSE-NOT: taskflow.task @consumer

// TILE-FUSE: taskflow.task @producer.tile.0.0.tile.1.0.fuse.consumer.tile.0.0.tile.1.0
// TILE-FUSE: taskflow.task @producer.tile.0.0.tile.1.1.fuse.consumer.tile.0.0.tile.1.1
// TILE-FUSE: taskflow.task @producer.tile.0.1.tile.1.0.fuse.consumer.tile.0.1.tile.1.0
// TILE-FUSE: taskflow.task @producer.tile.0.1.tile.1.1.fuse.consumer.tile.0.1.tile.1.1
// TILE-FUSE: taskflow.join

// FUSE-TILE: taskflow.task @producer.fuse.consumer.tile.0.0.tile.1.0
// FUSE-TILE: taskflow.task @producer.fuse.consumer.tile.0.0.tile.1.1
// FUSE-TILE: taskflow.task @producer.fuse.consumer.tile.0.1.tile.1.0
// FUSE-TILE: taskflow.task @producer.fuse.consumer.tile.0.1.tile.1.1
// FUSE-TILE: taskflow.join

// SEQ: taskflow.task @producer.k.0
// SEQ: taskflow.task @fused.producer.k.1.consumer
// SEQ-NOT: taskflow.task @consumer

// LINEAR: taskflow.task @partial_k.0
// LINEAR: taskflow.task @partial_k.1
// LINEAR: taskflow.task @fused.reduction.final.consumer
// LINEAR-NOT: taskflow.task @consumer

// TREE: taskflow.task @partial_k.0
// TREE: taskflow.task @partial_k.1
// TREE: taskflow.task @reduction.tree.0.0
// TREE: taskflow.task @fused.reduction.final.consumer

// REJECT-EARLY: sequential consumer fusion requires a completed final K producer
