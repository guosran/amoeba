//===- SpatialTaskCandidateSpace.h -----------------------------*- C++ -*-===//
//
// Defines the fixed-orientation spatial shape alphabet and its Cartesian
// traversal. Resource feasibility is intentionally deferred to the production
// spatial-temporal scheduler, where non-overlapping tasks may reuse CGRAs.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H
#define AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H

#include "AnalyticalTaskCandidateCommon.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr llvm::StringLiteral kShapePolicy =
    "static-oriented-rectangles";
inline constexpr llvm::StringLiteral kSearchScope =
    "static-shape-cartesian-product";
inline constexpr llvm::StringLiteral kSpatialCapacityPolicy =
    "per-task-fit-temporal-reuse-allowed";
inline constexpr llvm::StringLiteral kShapePruningPolicy = "none";
inline constexpr llvm::StringLiteral kSpatialCandidateBudgetPolicy =
    "fail-before-publish";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  int64_t tripCount = 1;
  RectShape shape;
};

// Stores one ordered shape choice for every Taskflow task.
struct Candidate {
  std::string id;
  llvm::SmallVector<TaskShapeChoice> choices;
};

using ShapeIndexTupleConsumer =
    llvm::function_ref<bool(uint64_t, llvm::ArrayRef<size_t>)>;

llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols,
                                                       int64_t maxCgrasPerTask);
// Visits the complete per-task shape Cartesian product. Each shape already
// fits the physical grid on its own; whole-program feasibility is determined
// later by the production spatial-temporal scheduler. `shapeIndices` follows
// task order and the candidate index is contiguous from zero.
bool visitShapeCartesianProduct(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    ShapeIndexTupleConsumer consume);
llvm::json::Object candidateJson(const Candidate &candidate);
} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H
