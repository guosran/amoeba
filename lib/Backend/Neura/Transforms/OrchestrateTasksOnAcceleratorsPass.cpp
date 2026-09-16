// Orchestrate Taskflow tasks onto a multi-CGRA grid.

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "Backend/Neura/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

std::unique_ptr<Orchestration>
createOrchestrationStrategy(StringRef strategy_name, int grid_rows,
                            int grid_cols, SchedulingMode mode,
                            bool communication_aware) {
  return llvm::StringSwitch<std::unique_ptr<Orchestration>>(strategy_name)
      .Case("routing-critical-path",
            std::make_unique<RoutingCriticalPathOrchestration>(grid_rows,
                                                               grid_cols, mode))
      .Case("analytical-based-task-orchestration",
            std::make_unique<AnalyticalBasedTaskOrchestration>(
                grid_rows, grid_cols, mode, communication_aware))
      .Default(nullptr);
}

struct OrchestrateTasksOnAcceleratorsPass
    : public PassWrapper<OrchestrateTasksOnAcceleratorsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OrchestrateTasksOnAcceleratorsPass)

  OrchestrateTasksOnAcceleratorsPass() = default;
  OrchestrateTasksOnAcceleratorsPass(
      const OrchestrateTasksOnAcceleratorsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "orchestrate-tasks-on-accelerators";
  }
  StringRef getDescription() const override {
    return "Orchestrates Taskflow tasks onto a 2D multi-CGRA grid (spatial or "
           "spatial-temporal)";
  }

  Option<std::string> schedulingMode{
      *this, "scheduling-mode",
      llvm::cl::desc("Task scheduling mode: 'spatial' (one task per CGRA, "
                     "asserts if tasks exceed the grid size) or "
                     "'spatial-temporal' (default, time-multiplexes CGRAs so "
                     "task count is not bounded by grid size)."),
      llvm::cl::init("spatial-temporal")};

  Option<std::string> orchestrationStrategy{
      *this, "orchestration-strategy",
      llvm::cl::desc("Task orchestration strategy: 'routing-critical-path' "
                     "(default) or 'analytical-based-task-orchestration' "
                     "(requires a materialized joint-scheduling candidate)."),
      llvm::cl::init("routing-critical-path")};

  Option<std::string> communicationMode{
      *this, "communication-mode",
      llvm::cl::desc("Task communication mode: 'none' or 'explicit'."),
      llvm::cl::init("none")};

  void runOnOperation() override {
    SchedulingMode mode;
    if (schedulingMode.getValue() == "spatial") {
      mode = SchedulingMode::Spatial;
    } else if (schedulingMode.getValue() == "spatial-temporal") {
      mode = SchedulingMode::SpatialTemporal;
    } else {
      getOperation()->emitError()
          << "unknown task scheduling mode: " << schedulingMode.getValue();
      signalPassFailure();
      return;
    }
    bool communication_aware = false;
    if (communicationMode.getValue() == "explicit") {
      communication_aware = true;
    } else if (communicationMode.getValue() != "none") {
      getOperation()->emitError()
          << "unknown communication mode: " << communicationMode.getValue();
      signalPassFailure();
      return;
    }
    if (communication_aware && orchestrationStrategy.getValue() !=
                                   "analytical-based-task-orchestration") {
      getOperation()->emitError()
          << "explicit communication requires joint resource binding";
      signalPassFailure();
      return;
    }
    const neura::Architecture &architecture = neura::getArchitecture();
    std::unique_ptr<Orchestration> strategy = createOrchestrationStrategy(
        orchestrationStrategy.getValue(), architecture.getMultiCgraRows(),
        architecture.getMultiCgraColumns(), mode, communication_aware);
    if (!strategy) {
      getOperation()->emitError() << "unknown task orchestration strategy: "
                                  << orchestrationStrategy.getValue();
      signalPassFailure();
      return;
    }

    if (!strategy->runTaskOrchestration(getOperation())) {
      getOperation()->emitError()
          << "failed to orchestrate taskflow tasks with strategy: "
          << orchestrationStrategy.getValue();
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createOrchestrateTasksOnAcceleratorsPass() {
  return std::make_unique<OrchestrateTasksOnAcceleratorsPass>();
}

} // namespace mlir::amoeba::neura
