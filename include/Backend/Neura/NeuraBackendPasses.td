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
    Neura multi-CGRA architecture. The current strategy prioritizes routing
    critical paths and supports spatial and spatial-temporal scheduling.
  }];
  let options = [Option<
      "schedulingMode", "scheduling-mode", "std::string",
      /*default=*/"\"spatial-temporal\"",
      "Task scheduling mode: 'spatial' (one task per CGRA) or "
      "'spatial-temporal' (time-multiplexes CGRAs).">];
  let constructor =
      "::mlir::amoeba::neura::createOrchestrateTasksOnAcceleratorsPass()";
}

def EnumerateAnalyticalTaskCandidates
    : Pass<"enumerate-analytical-task-candidates", "ModuleOp"> {
  let summary = "Freeze all concurrently packable task-shape candidates";
  let description = [{
    Enumerates every ordered assignment of fixed-orientation rectangular
    per-task CGRA shapes that admits a simultaneous, non-overlapping placement
    on the physical multi-CGRA grid, then writes an explicit JSONL manifest.
    Exact placement feasibility is checked with a small backtracking search;
    temporal reuse cannot make an over-capacity tuple legal.  The concrete
    placement coordinates remain a downstream heuristic choice.  The manifest
    records the explicit grid and per-CGRA tile dimensions read from
    architecture.yaml and assigns contiguous IDs such as candidate-0.  This
    pass does not consult an analytical score or invoke Neura's mapper. After
    the complete manifest is published, it annotates each source task with the
    canonical body hash that a task-DFG extractor must preserve. The initial
    implementation fixes fusion, fission, tiling, concrete placement, and
    temporal order; later search axes extend the same contract.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Path to the candidate JSONL manifest.">,
    Option<"maxCandidates", "max-candidates", "int64_t",
           /*default=*/"1000000",
           "Fail instead of publishing a partial manifest above this count.">,
    Option<"maxCgrasPerTask", "max-cgras-per-task", "int64_t",
           /*default=*/"4",
           "Largest rectangular physical footprint to enumerate per task.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ScoreAnalyticalTaskCandidates
    : Pass<"score-analytical-task-candidates", "ModuleOp"> {
  let summary = "Score every candidate in a frozen analytical-task manifest";
  let description = [{
    Validates and scores every candidate from a complete manifest using a
    named ML II/startup cost catalogue.  All repeated queries share one
    (task, mapper tile rows, mapper tile columns) cache.  Only after the
    complete manifest has been consumed is the top-k shortlist emitted.  The
    current shape-only score is explicitly a compute-bottleneck estimate, not
    the final communication-aware pipeline interval.  The catalogue uses
    explicit mapper tile dimensions and a fixed cost schema.
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
           "Number of candidates to shortlist; zero means every valid one.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createScoreAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeAnalyticalTaskCandidate
    : Pass<"materialize-analytical-task-candidate", "ModuleOp"> {
  let summary = "Materialize one frozen analytical task candidate";
  let description = [{
    Selects exactly one candidate by sequential ID (or by manifest index for
    testing) and writes its cgra_count/cgra_shape decisions to Taskflow tasks.
    It does not invoke the mapper.  An external driver may therefore run this
    pass followed by the unchanged heuristic mapper exactly once per top-k ID.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the frozen candidate JSONL manifest.">,
    Option<"candidateId", "candidate-id", "std::string", /*default=*/"\"\"",
           "Sequential candidate ID to materialize.">,
    Option<"candidateIndex", "candidate-index", "int64_t", /*default=*/"-1",
           "Zero-based manifest index; intended for tests/debugging.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeAnalyticalTaskCandidatePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
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
