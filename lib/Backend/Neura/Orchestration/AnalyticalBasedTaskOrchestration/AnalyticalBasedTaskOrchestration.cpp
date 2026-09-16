// Orchestrates a materialized joint-scheduling candidate.

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using llvm::SmallVector;

namespace mlir {
namespace taskflow {

SmallVector<CgraShape> AnalyticalBasedTaskOrchestration::getRectangularShapes(
    int cgra_count, int grid_rows, int grid_cols) {
  SmallVector<CgraShape> shapes;
  if (cgra_count <= 0 || grid_rows <= 0 || grid_cols <= 0) {
    return shapes;
  }

  for (int rows = 1; rows <= grid_rows; ++rows) {
    if (cgra_count % rows != 0) {
      continue;
    }
    int cols = cgra_count / rows;
    if (cols <= grid_cols) {
      shapes.push_back({rows, cols, true, {}});
    }
  }
  return shapes;
}

namespace {

// Validates the cgra_shape produced by the analytical candidate materializer
// before the scheduler can consume an incomplete resource assignment.
bool validateShapeString(StringRef shape_text, int cgra_count, int grid_rows,
                         int grid_cols, std::string &error) {
  size_t x_pos = shape_text.find('x');
  if (x_pos == StringRef::npos || x_pos == 0 ||
      x_pos + 1 >= shape_text.size() ||
      shape_text.find('x', x_pos + 1) != StringRef::npos) {
    error = "cgra_shape must be a fixed rectangle of the form 'rowsxcols'";
    return false;
  }

  int rows = 0;
  int cols = 0;
  if (shape_text.take_front(x_pos).getAsInteger(10, rows) || rows <= 0 ||
      shape_text.drop_front(x_pos + 1).getAsInteger(10, cols) || cols <= 0) {
    error = "cgra_shape has invalid rectangle dimensions";
    return false;
  }
  if (rows > grid_rows || cols > grid_cols) {
    error = "cgra_shape rectangle does not fit the CGRA grid";
    return false;
  }
  if (static_cast<int64_t>(rows) * cols != cgra_count) {
    error = "cgra_shape area does not equal cgra_count";
    return false;
  }
  return true;
}
} // namespace

bool AnalyticalBasedTaskOrchestration::validateFixedShapeAttributes(
    func::FuncOp func) const {
  auto candidate_id =
      func->getAttrOfType<StringAttr>("joint_scheduling_candidate_id");
  if (!candidate_id || candidate_id.getValue().empty()) {
    func.emitError() << "joint resource binding requires a non-empty "
                        "joint_scheduling_candidate_id attribute";
    return false;
  }

  bool valid = true;
  func.walk([&](TaskflowTaskOp task) {
    if (!valid) {
      return;
    }
    if (!task->getAttrOfType<UnitAttr>(
            "amoeba.joint_shape_orientation_fixed")) {
      task.emitError() << "joint resource binding requires "
                          "amoeba.joint_shape_orientation_fixed";
      valid = false;
      return;
    }

    auto cgra_count = task->getAttrOfType<IntegerAttr>("cgra_count");
    if (!cgra_count || cgra_count.getInt() <= 0 ||
        cgra_count.getInt() > std::numeric_limits<int>::max()) {
      task.emitError() << "joint resource binding requires a "
                          "positive cgra_count";
      valid = false;
      return;
    }

    auto cgra_shape = task->getAttrOfType<StringAttr>("cgra_shape");
    if (!cgra_shape || cgra_shape.getValue().empty()) {
      task.emitError() << "joint resource binding requires a "
                          "non-empty cgra_shape";
      valid = false;
      return;
    }

    std::string error;
    if (!validateShapeString(cgra_shape.getValue(),
                             static_cast<int>(cgra_count.getInt()), grid_rows_,
                             grid_cols_, error)) {
      task.emitError() << "invalid fixed cgra_shape: " << error;
      valid = false;
    }
  });
  return valid;
}

void AnalyticalBasedTaskOrchestration::addDependencyEdge(
    TaskSuccessorMap &successors, Operation *producer,
    Operation *consumer) const {
  if (!producer || !consumer || producer == consumer) {
    return;
  }
  SmallVector<Operation *> &producer_successors = successors[producer];
  if (!llvm::is_contained(producer_successors, consumer)) {
    producer_successors.push_back(consumer);
  }
}

int AnalyticalBasedTaskOrchestration::computeDependencyDepth(
    Operation *task, TaskSuccessorMap &successors,
    DenseMap<Operation *, int> &depth_cache,
    DenseSet<Operation *> &visiting) const {
  if (auto it = depth_cache.find(task); it != depth_cache.end()) {
    return it->second;
  }
  if (!visiting.insert(task).second) {
    return 0;
  }

  int max_child_depth = 0;
  for (Operation *successor : successors[task]) {
    max_child_depth = std::max(
        max_child_depth,
        computeDependencyDepth(successor, successors, depth_cache, visiting) +
            1);
  }

  visiting.erase(task);
  depth_cache[task] = max_child_depth;
  return max_child_depth;
}

TaskPriorityMap
AnalyticalBasedTaskOrchestration::computeRoutingCriticalPathPriority(
    func::FuncOp func) const {
  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  TaskSuccessorMap successors;
  for (TaskflowTaskOp task : tasks) {
    (void)successors[task.getOperation()];
  }

  for (TaskflowTaskOp consumer : tasks) {
    Operation *consumer_op = consumer.getOperation();
    auto add_producer_from_value = [&](Value value) {
      DenseSet<Operation *> visited_channels;
      while (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
        if (!visited_channels.insert(channel.getOperation()).second)
          return;
        value = channel.getSource();
      }
      if (auto producer = value.getDefiningOp<TaskflowTaskOp>()) {
        addDependencyEdge(successors, producer.getOperation(), consumer_op);
      }
    };

    for (Value value_input : consumer.getValueInputs()) {
      add_producer_from_value(value_input);
    }
    for (Value operand : consumer->getOperands()) {
      add_producer_from_value(operand);
    }
  }

  TaskPriorityMap priority;
  DenseMap<Operation *, int> depth_cache;
  DenseSet<Operation *> visiting;
  for (TaskflowTaskOp task : tasks) {
    Operation *task_op = task.getOperation();
    priority[task_op] =
        computeDependencyDepth(task_op, successors, depth_cache, visiting);
  }
  return priority;
}

bool AnalyticalBasedTaskOrchestration::runTaskOrchestration(func::FuncOp func) {
  if (!validateFixedShapeAttributes(func)) {
    return false;
  }

  TaskPriorityMap priority = computeRoutingCriticalPathPriority(func);
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_,
                          ShapeSelectionPolicy::FixedOrientation);
  if (!scheduler.schedule(func, priority)) {
    return false;
  }
  func->setAttr("joint_scheduling_predicted_makespan",
                IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                 scheduler.getScheduleMakespan()));
  func.walk([](TaskflowTaskOp task) {
    task->removeAttr("amoeba.joint_shape_orientation_fixed");
  });
  return true;
}

} // namespace taskflow
} // namespace mlir
