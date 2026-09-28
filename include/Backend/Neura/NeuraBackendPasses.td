// NeuraBackendPasses.td - Passes owned by the Neura backend

#ifndef AMOEBA_BACKEND_NEURA_PASSES_TD
#define AMOEBA_BACKEND_NEURA_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp"> {
  let summary = "Convert Taskflow operations to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations and their counters into
    neura.kernel operations suitable for CGRA tile-array mapping.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConvertTaskflowToNeuraPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::affine::AffineDialect"];
}

def ConstructHyperblockFromTask
    : Pass<"construct-hyperblock-from-task", "func::FuncOp"> {
  let summary = "Construct Taskflow hyperblocks for the Neura backend";
  let description = [{
    Selects one of two transforms based on the Neura architecture.
    With counter support, the pass detects perfect nested loop bands.
    Without counter support, it wraps the innermost loop as a hyperblock.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConstructHyperblockFromTaskPass()";
}

def ClassifyTaskAndCounter : Pass<"classify-task-and-counter", "ModuleOp"> {
  let summary = "Classify tasks and counters for Neura runtime management";
  let description = [{
    Annotates taskflow.task and taskflow.counter operations with the
    structural and runtime-management information consumed by Neura.
  }];
  let constructor =
      "::mlir::amoeba::neura::createClassifyTaskAndCounterPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def OrchestrateTasksOnAccelerators
    : Pass<"orchestrate-tasks-on-accelerators", "func::FuncOp"> {
  let summary = "Orchestrate Taskflow tasks on a Neura multi-CGRA grid";
  let description = [{
    Computes task resource allocation, placement, and scheduling for the
    Neura multi-CGRA architecture. Selects an orchestration strategy and
    invokes the shared scheduler. The routing-critical-path strategy is the
    default. The analytical-based-task-orchestration strategy consumes a
    materialized task candidate and preserves its spatial shape orientation.
  }];
  let options = [Option<
      "schedulingMode", "scheduling-mode", "std::string",
      /*default=*/"\"spatial-temporal\"",
      "Task scheduling mode: 'spatial' (one task per CGRA) or "
      "'spatial-temporal' (time-multiplexes CGRAs).">,
    Option<
      "orchestrationStrategy", "orchestration-strategy", "std::string",
      /*default=*/"\"routing-critical-path\"",
      "Task orchestration strategy: 'routing-critical-path' (default) or "
      "'analytical-based-task-orchestration' (requires a materialized "
      "joint-scheduling candidate).">,
    Option<
      "communicationMode", "communication-mode", "std::string",
      /*default=*/"\"none\"",
      "Task communication mode: 'none' or strict architecture-level "
      "'explicit'.">];
  let constructor =
      "::mlir::amoeba::neura::createOrchestrateTasksOnAcceleratorsPass()";
}

def AnalyzeRecResMii : Pass<"analyze-rec-res-mii", "ModuleOp"> {
  let summary = "Compute mapper-compatible RecMII and ResMII without mapping";
  let description = [{
    Computes the recurrence and resource bounds for exactly one standalone
    task DFG. x-tiles and y-tiles select the concrete mapper rectangle. The
    pinned ensemble predictor reads amoeba.rec_mii and amoeba.res_mii as
    module attributes. The pass never places or routes an operation.
  }];
  let options = [Option<"xTiles", "x-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile columns; must be positive.">,
                 Option<"yTiles", "y-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile rows; must be positive.">];
  let constructor =
      "::mlir::amoeba::neura::createAnalyzeRecResMiiPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect"];
}

def EnumerateAnalyticalTaskCandidates
    : Pass<"enumerate-analytical-task-candidates", "ModuleOp"> {
  let summary = "Freeze the full static task-shape Cartesian product";
  let description = [{
    Enumerates every ordered assignment of fixed-orientation rectangular
    per-task CGRA shapes that individually fit the physical grid, then writes
    a JSONL manifest. The production spatial-temporal scheduler determines
    whole-program feasibility and may reuse resources over time. Constant counter chains
    record their exact trip count, while dynamic or unresolved bounds are
    rejected. Counter bounds are read directly from their SSA definitions, so
    this pass does not depend on `classify-task-and-counter`. The pass does not
    score candidates or invoke the mapper. Fusion, fission, tiling, placement,
    and temporal order are fixed.
  }];
  let options =
      [Option<
           "functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
       Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
              "Path to the candidate JSONL manifest.">,
       Option<
           "maxCandidates", "max-candidates", "int64_t",
           /*default=*/"1000000",
           "Fail instead of publishing a partial manifest above this count.">,
       Option<"maxCgrasPerTask", "max-cgras-per-task", "int64_t",
              /*default=*/"0",
              "Largest rectangular physical footprint per task; zero uses the full physical grid.">,
       Option<"graphVariantId", "graph-variant-id", "std::string",
              /*default=*/"\"\"", "Optional graph variant identity.">,
       Option<"graphVariantSha256", "graph-variant-sha256", "std::string",
              /*default=*/"\"\"", "SHA-256 of graph-variant result IR bytes.">];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeAnalyticalTaskCandidate
    : Pass<"materialize-analytical-task-candidate", "ModuleOp"> {
  let summary = "Materialize one static task-shape candidate";
  let description = [{
    Validates a complete candidate manifest, selects one candidate by its
    sequential ID, and writes its cgra_count and cgra_shape decisions to
    Taskflow tasks. The analytical-based task orchestration strategy consumes
    and preserves the selected rectangle orientation. This pass does not
    invoke the mapper. Counter annotations must match the enumeration pipeline
    that produced the manifest.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the candidate JSONL manifest.">,
    Option<"candidateId", "candidate-id", "std::string", /*default=*/"\"\"",
           "Sequential candidate ID to materialize.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeAnalyticalTaskCandidatePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MapJointSchedulingTasks
    : Pass<"map-joint-scheduling-tasks", "ModuleOp"> {
  let summary = "Map every selected task on its fixed oriented rectangle";
  let description = [{
    Runs InsertDataMov and the real Neura mapper independently for every
    materialized task. The selected per-task mapper rows and columns configure
    each invocation. Mapping failure, missing kernels, or a mismatched result
    fails closed. Successfully mapped operations are moved back into the task.
  }];
  let constructor =
      "::mlir::amoeba::neura::createMapJointSchedulingTasksPass()";
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Unique Taskflow function selected by the manifest.">,
    Option<"scoreFile", "scores", "std::string", /*default=*/"\"\"",
           "Validated score JSONL used to bind startup cycles.">,
    Option<"candidateId", "candidate-id", "std::string",
           /*default=*/"\"\"", "Selected candidate ID.">,
    Option<"expectedTraceSha256", "expected-trace-sha256", "std::string",
           /*default=*/"\"\"", "Expected selected schedule trace identity.">,
    Option<"expectedScoreSha256", "expected-score-sha256", "std::string",
           /*default=*/"\"\"", "SHA-256 of the exact score JSONL bytes.">
  ];
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ScoreAnalyticalTaskCandidates
    : Pass<"score-analytical-task-candidates", "ModuleOp"> {
  let summary = "Score every candidate in a frozen spatial-binding manifest";
  let description = [{
    Validates and scores every candidate from a complete manifest using a
    provenance-bound external ML II/startup cost catalogue. Repeated queries
    share a stable task-body, architecture, and oriented-shape cache. Only
    after complete manifest traversal is a deterministic top-k shortlist
    emitted. The ranking score is the makespan produced by the production
    fixed-orientation spatial-temporal scheduler using predicted task
    durations. Communication remains outside this pass. A symbol-dynamic trip
    count must be concretely bound before ranking.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the frozen candidate JSONL manifest.">,
    Option<"costFile", "cost-file", "std::string", /*default=*/"\"\"",
           "Path to the task-shape ML cost catalogue.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Path to the score JSONL manifest.">,
    Option<"topK", "top-k", "int64_t", /*default=*/"1",
           "Candidates to shortlist; zero selects every valid candidate.">,
    Option<"communicationMode", "communication-mode", "std::string",
           /*default=*/"\"none\"",
           "Task communication mode: 'none' or strict 'explicit'.">,
    Option<"graphVariantId", "graph-variant-id", "std::string",
           /*default=*/"\"\"", "Optional graph variant identity.">,
    Option<"graphVariantSha256", "graph-variant-sha256", "std::string",
           /*default=*/"\"\"", "SHA-256 of graph-variant result IR bytes.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createScoreAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary = "Fuse Taskflow tasks using Neura profitability metrics";
  let description = [{
    Applies producer-consumer and sibling fusion strategies. Profitability is
    evaluated with Neura lowering and MII metrics.
  }];
  let constructor = "::mlir::amoeba::neura::createFuseTaskPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ExtractJointTaskGraphFacts
    : Pass<"extract-joint-task-graph-facts", "ModuleOp"> {
  let summary = "Extract Taskflow graph facts and legal rewrite actions";
  let description = [{
    Writes deterministic task facts, dependencies, shape needs, a canonical
    structural key, and conservative fusion, fission, and tensor-axis tiling
    actions. Graph enumeration reruns this pass after every rewrite.
  }];
  let constructor =
      "::mlir::amoeba::neura::createExtractJointTaskGraphFactsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def EnumerateJointSemanticRewrite
    : Pass<"enumerate-joint-semantic-rewrites", "ModuleOp"> {
  let summary = "Enumerate source-bound Taskflow tiling and fusion graph rewrites";
  let description = [{
    Recognizes a static producer-consumer Taskflow program and publishes a
    deterministic, hardware-independent graph rewrite DAG. Legalization and
    candidate identity are owned by this MLIR pass.
  }];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateJointSemanticRewritePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeJointSemanticRewrite
    : Pass<"materialize-joint-semantic-rewrite", "ModuleOp"> {
  let summary = "Materialize a selected source-bound Taskflow graph rewrite";
  let description = [{
    Applies a selected hardware-independent M/N/K task tiling and graph
    fusion candidate to Taskflow IR after source and legality validation.
  }];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeJointSemanticRewritePass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileOutputM : Pass<"tile-output-m", "ModuleOp"> {
  let summary = "Tile a source-bound Taskflow output along M";
  let constructor = "::mlir::amoeba::neura::createTileOutputMPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileOutputN : Pass<"tile-output-n", "ModuleOp"> {
  let summary = "Tile a source-bound Taskflow output along N";
  let constructor = "::mlir::amoeba::neura::createTileOutputNPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKSequential
    : Pass<"tile-reduction-k-sequential", "ModuleOp"> {
  let summary = "Split K into sequential Taskflow state-carry blocks";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKSequentialPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKParallelLinear
    : Pass<"tile-reduction-k-parallel-linear", "ModuleOp"> {
  let summary = "Split K into private partials with linear reduction topology";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKParallelLinearPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileReductionKParallelTree
    : Pass<"tile-reduction-k-parallel-tree", "ModuleOp"> {
  let summary = "Split K into private partials with balanced-tree topology";
  let constructor =
      "::mlir::amoeba::neura::createTileReductionKParallelTreePass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseProducerConsumer
    : Pass<"fuse-producer-consumer", "ModuleOp"> {
  let summary = "Fuse a proven Taskflow producer-consumer pair";
  let constructor =
      "::mlir::amoeba::neura::createFuseProducerConsumerPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseTileLocalPair : Pass<"fuse-tile-local-pair", "ModuleOp"> {
  let summary = "Fuse corresponding producer-consumer output tiles";
  let constructor = "::mlir::amoeba::neura::createFuseTileLocalPairPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseReductionConsumer
    : Pass<"fuse-reduction-consumer", "ModuleOp"> {
  let summary = "Fuse a completed numeric reduction with its consumer";
  let constructor =
      "::mlir::amoeba::neura::createFuseReductionConsumerPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeNumericReduction
    : Pass<"materialize-numeric-reduction", "ModuleOp"> {
  let summary = "Materialize a numeric reduction over private K partials";
  let constructor =
      "::mlir::amoeba::neura::createMaterializeNumericReductionPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeCompletionJoin
    : Pass<"materialize-completion-join", "ModuleOp"> {
  let summary = "Join disjoint Taskflow completion states without arithmetic";
  let constructor =
      "::mlir::amoeba::neura::createMaterializeCompletionJoinPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def LowerJointTaskflowToHostSCF
    : Pass<"lower-joint-taskflow-to-host-scf", "ModuleOp"> {
  let summary = "Lower executable ORBIT Taskflow graphs to host SCF";
  let description = [{
    Inlines dependency-ordered Taskflow bodies, lowers static or dynamic
    counters to nested scf.for operations, and aliases completion joins only
    after the Taskflow verifier proves their region coverage. Unsupported
    bodies and operations are rejected without publishing a partial lowering.
  }];
  let options = [Option<
      "functionName", "function", "std::string", /*default=*/"\"\"",
      "Only lower this function; empty lowers every Taskflow function.">];
  let constructor =
      "::mlir::amoeba::neura::createLowerJointTaskflowToHostSCFPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FissionTask : Pass<"fission-task", "func::FuncOp"> {
  let summary = "Split one value-only task at a legal pure DFG cut";
  let constructor = "::mlir::amoeba::neura::createFissionTaskPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TileTask : Pass<"tile-task", "func::FuncOp"> {
  let summary = "Split a static tensor task into k parts along one axis";
  let constructor = "::mlir::amoeba::neura::createTileTaskPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ResourceAwareTaskOptimization
    : Pass<"resource-aware-task-optimization", "func::FuncOp"> {
  let summary =
      "Optimize Taskflow resource allocation for the Neura CGRA backend";
  let description = [{
    Balances pipeline latency and fuses independent tasks for Neura CGRA
    utilization. Compiled estimation runs the Neura lowering and mapping
    flow; analytical estimation uses ResMII/RecMII estimates.
  }];
  let options = [Option<
      "estimationMode", "estimation-mode", "std::string",
      /*default=*/"\"compiled\"",
      "Profiling mode: 'compiled' runs Neura lowering and mapping; "
      "'analytical' uses ResMII/RecMII estimates.">];
  let constructor =
      "::mlir::amoeba::neura::createResourceAwareTaskOptimizationPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

#endif // AMOEBA_BACKEND_NEURA_PASSES_TD
