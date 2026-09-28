//===- TaskGraphRewriteLegality.h -----------------------------*- C++ -*-===//
//
// Shared fail-closed legality helpers for task-graph rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_TASK_GRAPH_REWRITE_LEGALITY_H
#define AMOEBA_JOINT_SCHEDULING_TASK_GRAPH_REWRITE_LEGALITY_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline bool isNoAliasTaskFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

inline bool isTaskFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front();
}

// Rewrites replace task symbols with one or more derived tasks.  Until the
// control-edge contract has an explicit name-remapping rule, a task that is
// named by any control predecessor list cannot be rewritten safely.
inline bool isNamedControlPredecessor(taskflow::TaskflowTaskOp task) {
  auto function = task->getParentOfType<func::FuncOp>();
  if (!function)
    return true;
  bool referenced = false;
  function.walk([&](taskflow::TaskflowTaskOp consumer) {
    Attribute attribute = consumer->getAttr("amoeba.control_predecessors");
    if (!attribute)
      return;
    auto names = dyn_cast<ArrayAttr>(attribute);
    if (!names) {
      referenced = true;
      return;
    }
    for (Attribute nameAttribute : names) {
      StringRef name;
      if (auto string = dyn_cast<StringAttr>(nameAttribute))
        name = string.getValue();
      else if (auto symbol = dyn_cast<FlatSymbolRefAttr>(nameAttribute))
        name = symbol.getValue();
      else {
        referenced = true;
        return;
      }
      referenced |= name == task.getTaskName();
    }
  });
  return referenced;
}

inline bool provesDistinctTaskStorage(Value lhs, Value rhs) {
  if (lhs == rhs)
    return false;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  bool lhsAllocation =
      lhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(lhsDefinition);
  bool rhsAllocation =
      rhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(rhsDefinition);
  if (lhsAllocation && (rhsAllocation || isTaskFunctionArgument(rhs)))
    return true;
  if (rhsAllocation && isTaskFunctionArgument(lhs))
    return true;
  return isNoAliasTaskFunctionArgument(lhs) &&
         isNoAliasTaskFunctionArgument(rhs);
}

inline bool mayAliasTaskStorage(Value lhs, Value rhs) {
  return lhs == rhs || !provesDistinctTaskStorage(lhs, rhs);
}

// Moving `moved` from before `crossed` to after it reverses their observable
// memory order.  Incomplete original-memory summaries are deliberately treated
// as a conflict: SSA task state alone is not sufficient because Taskflow also
// permits direct accesses to the same original memref.
inline bool
hasConflictingOriginalMemoryEffects(taskflow::TaskflowTaskOp moved,
                                    taskflow::TaskflowTaskOp crossed) {
  if (moved.getOriginalReadMemrefs().size() != moved.getWillReads().size() ||
      moved.getOriginalWriteMemrefs().size() != moved.getWillWrites().size() ||
      crossed.getOriginalReadMemrefs().size() !=
          crossed.getWillReads().size() ||
      crossed.getOriginalWriteMemrefs().size() !=
          crossed.getWillWrites().size())
    return true;

  for (Value movedRead : moved.getOriginalReadMemrefs())
    for (Value crossedWrite : crossed.getOriginalWriteMemrefs())
      if (mayAliasTaskStorage(movedRead, crossedWrite))
        return true;
  for (Value movedWrite : moved.getOriginalWriteMemrefs()) {
    for (Value crossedRead : crossed.getOriginalReadMemrefs())
      if (mayAliasTaskStorage(movedWrite, crossedRead))
        return true;
    for (Value crossedWrite : crossed.getOriginalWriteMemrefs())
      if (mayAliasTaskStorage(movedWrite, crossedWrite))
        return true;
  }
  return false;
}

inline bool isKnownFissionBlockArgument(Value value,
                                        taskflow::TaskflowTaskOp task,
                                        Block &hyperblockBody) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  if (argument.getOwner() == &hyperblockBody)
    return true;
  if (argument.getOwner() != &task.getBody().front())
    return false;
  size_t expected = task.getWillReads().size() + task.getWillWrites().size() +
                    task.getValueInputs().size();
  return argument.getArgNumber() < expected;
}

// Prefix fission tasks carry the source reads and scalar inputs, but not the
// externally visible output state.  A selected prefix node that captures a
// will_writes block argument would retain a dangling reference after the source
// task is erased.
inline bool isAvailableInFissionPrefix(Value value,
                                       taskflow::TaskflowTaskOp task,
                                       Block &hyperblockBody) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return true;
  if (argument.getOwner() == &hyperblockBody)
    return true;
  if (argument.getOwner() != &task.getBody().front())
    return false;
  size_t index = argument.getArgNumber();
  size_t readCount = task.getWillReads().size();
  size_t writeEnd = readCount + task.getWillWrites().size();
  size_t expected = writeEnd + task.getValueInputs().size();
  return index < readCount || (index >= writeEnd && index < expected);
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_JOINT_SCHEDULING_TASK_GRAPH_REWRITE_LEGALITY_H
