//===- SpatialTaskCandidateSpace.cpp -----------------------------------===//
//
// Implements construction and traversal of the static rectangular
// task-shape candidate space.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::formatv("{0}x{1}", rows, cols);
  return result;
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The deterministic order defines
// the mixed-radix alphabet for candidate IDs.
// TODO: Extend the analytical candidate schema and its consumers to represent
// non-rectangular shapes. The analytical search intentionally enumerates only
// fixed-orientation rectangles until that contract exists end to end.
SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                 int64_t gridCols,
                                                 int64_t perCgraRows,
                                                 int64_t perCgraCols,
                                                 int64_t maxCgrasPerTask) {
  // The candidate domain contains only concrete integer rectangles.
  SmallVector<RectShape> result;
  if (gridRows <= 0 || gridCols <= 0 || perCgraRows <= 0 || perCgraCols <= 0 ||
      maxCgrasPerTask <= 0)
    return result;

  const int64_t gridSize =
      gridRows > std::numeric_limits<int64_t>::max() / gridCols
          ? std::numeric_limits<int64_t>::max()
          : gridRows * gridCols;
  for (int64_t count = 1; count <= std::min(gridSize, maxCgrasPerTask);
       ++count) {
    for (const CgraShape &physicalShape :
         taskflow::AnalyticalBasedTaskOrchestration::getRectangularShapes(
             static_cast<int>(count), static_cast<int>(gridRows),
             static_cast<int>(gridCols))) {
      if (physicalShape.rows >
              std::numeric_limits<int64_t>::max() / perCgraRows ||
          physicalShape.cols >
              std::numeric_limits<int64_t>::max() / perCgraCols)
        continue;
      int64_t mapperRows = physicalShape.rows * perCgraRows;
      int64_t mapperCols = physicalShape.cols * perCgraCols;
      result.push_back(
          {physicalShape.rows, physicalShape.cols, mapperRows, mapperCols});
    }
  }
  return result;
}

bool visitShapeCartesianProduct(ArrayRef<SmallVector<RectShape>> shapesByTask,
                                ShapeIndexTupleConsumer consume) {
  if (shapesByTask.empty() ||
      llvm::any_of(shapesByTask,
                   [](const auto &shapes) { return shapes.empty(); }))
    return true;

  SmallVector<size_t> selectedIndices;
  uint64_t index = 0;
  std::function<bool(size_t)> visit = [&](size_t taskIndex) {
    if (taskIndex == shapesByTask.size()) {
      if (!consume(index, selectedIndices))
        return false;
      if (index == std::numeric_limits<uint64_t>::max())
        return false;
      ++index;
      return true;
    }

    for (auto [shapeIndex, unused] : llvm::enumerate(shapesByTask[taskIndex])) {
      (void)unused;
      selectedIndices.push_back(shapeIndex);
      if (!visit(taskIndex + 1))
        return false;
      selectedIndices.pop_back();
    }
    return true;
  };
  return visit(0);
}

// Serializes the shape fields used by candidate records. The explicit tile
// dimensions are the only mapper-shape truth; a `rect-4x8` string is produced
// only for diagnostics when needed.
static llvm::json::Object shapeJson(const RectShape &shape) {
  llvm::json::Object object;
  object["kind"] = "rect";
  object["rows"] = shape.rows;
  object["cols"] = shape.cols;
  object["cgra_count"] = shape.cgraCount();
  object["cgra_shape"] = shape.toCgraShapeAttrValue();
  object["mapper_tile_rows"] = shape.mapperRows;
  object["mapper_tile_cols"] = shape.mapperCols;
  return object;
}

// Serializes one candidate while preserving task order. The candidate ID is a
// sequential index, so no task-body identity or file-derived metadata is
// required to interpret it within its validated manifest.
llvm::json::Object candidateJson(const Candidate &candidate) {
  llvm::json::Array choices;
  for (const TaskShapeChoice &choice : candidate.choices) {
    llvm::json::Object record;
    record["task"] = choice.task;
    record["trip_count"] = choice.tripCount;
    record["shape"] = shapeJson(choice.shape);
    choices.push_back(std::move(record));
  }
  llvm::json::Object record;
  record["record_type"] = "candidate";
  record["schema"] = kCandidateSchema.str();
  record["candidate_id"] = candidate.id;
  record["task_shapes"] = std::move(choices);
  return record;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
