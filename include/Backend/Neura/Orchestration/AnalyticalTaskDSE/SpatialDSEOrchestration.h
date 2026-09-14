// Orchestration of a materialized spatial analytical DSE candidate.

#ifndef AMOEBA_SPATIAL_DSE_ORCHESTRATION_H
#define AMOEBA_SPATIAL_DSE_ORCHESTRATION_H

#include "Backend/Neura/Orchestration/Orchestration.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
namespace taskflow {

// Concrete orchestration strategy for a materialized analytical candidate.
//
// The materializer records one complete resource decision on the function and
// each task.  This strategy validates that decision and asks TaskScheduler to
// place each task using the recorded orientation exactly as written.
class SpatialDSEOrchestration : public Orchestration {
public:
  SpatialDSEOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      SchedulingMode mode = SchedulingMode::SpatialTemporal)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {}

  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  std::string getName() const override { return "analytical-dse-spatial"; }

private:
  using TaskSuccessorMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

  bool validateFixedShapeAttributes(func::FuncOp func) const;

  void addDependencyEdge(TaskSuccessorMap &successors, Operation *producer,
                         Operation *consumer) const;

  int computeDependencyDepth(Operation *task, TaskSuccessorMap &successors,
                             llvm::DenseMap<Operation *, int> &depth_cache,
                             llvm::DenseSet<Operation *> &visiting) const;

  TaskPriorityMap computeRoutingCriticalPathPriority(func::FuncOp func) const;

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_SPATIAL_DSE_ORCHESTRATION_H
