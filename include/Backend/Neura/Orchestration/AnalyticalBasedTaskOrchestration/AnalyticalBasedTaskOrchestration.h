// Orchestration of a materialized joint-scheduling candidate.

#ifndef AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H
#define AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H

#include "Backend/Neura/Orchestration/Orchestration.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace taskflow {

// Orchestrates a materialized joint-scheduling candidate.
//
// The current candidate space varies each task's fixed-orientation rectangular
// region. The existing scheduler may execute the materialized candidate in
// spatial or spatial-temporal mode. Temporal candidate exploration is added
// separately.
class AnalyticalBasedTaskOrchestration : public Orchestration {
public:
  AnalyticalBasedTaskOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      SchedulingMode mode = SchedulingMode::SpatialTemporal)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {}

  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  // Enumerates legal fixed-orientation rectangular shapes for one CGRA count.
  static llvm::SmallVector<CgraShape>
  getRectangularShapes(int cgra_count, int grid_rows = kCgraGridRows,
                       int grid_cols = kCgraGridCols);

  // Reuse the production task order when estimating a candidate's makespan.
  TaskPriorityMap computeRoutingCriticalPathPriority(func::FuncOp func) const;

  std::string getName() const override {
    return "analytical-based-task-orchestration";
  }

private:
  using TaskSuccessorMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

  bool validateFixedShapeAttributes(func::FuncOp func) const;

  void addDependencyEdge(TaskSuccessorMap &successors, Operation *producer,
                         Operation *consumer) const;

  int computeDependencyDepth(Operation *task, TaskSuccessorMap &successors,
                             llvm::DenseMap<Operation *, int> &depth_cache,
                             llvm::DenseSet<Operation *> &visiting) const;

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H
