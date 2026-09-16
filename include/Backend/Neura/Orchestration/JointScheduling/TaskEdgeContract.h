//===- TaskEdgeContract.h ------------------------------------*- C++ -*-===//
//
// Public typed task-edge and payload contracts for joint scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_TASK_EDGE_CONTRACT_H
#define AMOEBA_JOINT_SCHEDULING_TASK_EDGE_CONTRACT_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// The attribute is intentionally outside the Taskflow dialect.  It is an
// optional contract consumed by analytical orchestration and does not change
// the semantics of the Taskflow operation itself.  Values are task-name
// strings in an ArrayAttr, in the order in which the control edges are
// declared.
inline constexpr llvm::StringLiteral kControlPredecessorsAttr =
    "amoeba.control_predecessors";

// Optional DenseI64ArrayAttr on a consumer task.  Its entries are laid out as
// [will_reads..., value_inputs...], matching the operand segment indices.  A
// non-negative entry is an explicit payload size in bits; -1 records that the
// size remains unknown.  The attribute is required to have exactly one entry
// for every data-carrying operand when present.
inline constexpr llvm::StringLiteral kEdgePayloadBitsAttr =
    "amoeba.edge_payload_bits";

// A task edge's semantic role.  Value and Raw edges carry data; the remaining
// kinds only impose execution order.
enum class TaskEdgeKind : uint8_t {
  Value,
  Raw,
  War,
  Waw,
  Control,
};

// The operand/result segment that identifies an edge endpoint.  None is used
// for the producer side of an explicit control edge, which has no SSA result.
enum class TaskOperandSegment : uint8_t {
  WillReads,
  WillWrites,
  ValueInputs,
  Control,
};

enum class TaskResultSegment : uint8_t {
  None,
  DoneReads,
  DoneWrites,
  ValueOutputs,
};

// One typed edge in the task graph.  Taskflow operation handles are used for
// graph integration, while segment and index fields make the edge identity
// stable and prevent distinct edges between the same task pair from being
// collapsed.
struct TaskEdge {
  taskflow::TaskflowTaskOp producer;
  taskflow::TaskflowTaskOp consumer;
  TaskEdgeKind kind = TaskEdgeKind::Control;
  TaskResultSegment producer_segment = TaskResultSegment::None;
  uint32_t producer_index = 0;
  TaskOperandSegment consumer_segment = TaskOperandSegment::Control;
  uint32_t consumer_index = 0;

  // A missing payload is an explicit unknown.  It must not be interpreted as
  // zero by a communication-aware scheduler.
  std::optional<uint64_t> payload_bits;
};

// Options controlling construction of a typed graph.  The default preserves
// graph inspection for callers that only need ordering.  Explicit
// communication mode should set require_payload, causing dynamic or otherwise
// unknown data sizes and channel sources without a task producer to fail
// closed.  A consumer's kEdgePayloadBitsAttr can provide validated facts for
// dynamic types without changing the IR type.
struct TaskEdgeGraphOptions {
  bool require_payload = false;
  bool read_control_predecessors = true;
};

// Immutable graph view returned by buildTaskEdgeGraph().
class TaskEdgeGraph {
public:
  llvm::ArrayRef<taskflow::TaskflowTaskOp> getTasks() const {
    return {tasks_.data(), tasks_.size()};
  }
  llvm::ArrayRef<TaskEdge> getEdges() const {
    return {edges_.data(), edges_.size()};
  }

private:
  friend FailureOr<TaskEdgeGraph>
  buildTaskEdgeGraph(func::FuncOp func, const TaskEdgeGraphOptions &options,
                     std::string &error);

  llvm::SmallVector<taskflow::TaskflowTaskOp> tasks_;
  llvm::SmallVector<TaskEdge> edges_;
};

// Builds typed edges from Taskflow operand/result segments.  Direct task SSA
// values and chains of taskflow.channel operations are supported.  Malformed
// segment connections and ambiguous channel chains fail instead of becoming
// an implicit zero-cost dependency.
FailureOr<TaskEdgeGraph> buildTaskEdgeGraph(func::FuncOp func,
                                            const TaskEdgeGraphOptions &options,
                                            std::string &error);

// Returns the statically known number of bits represented by a value type.
// Dynamic shapes, index values, and types without a fixed bit width return
// std::nullopt. A memory edge therefore needs an explicit runtime payload fact
// before strict communication scheduling.
std::optional<uint64_t> getStaticPayloadBits(Type type);

llvm::StringRef stringifyTaskEdgeKind(TaskEdgeKind kind);
llvm::StringRef stringifyTaskOperandSegment(TaskOperandSegment segment);
llvm::StringRef stringifyTaskResultSegment(TaskResultSegment segment);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_JOINT_SCHEDULING_TASK_EDGE_CONTRACT_H
