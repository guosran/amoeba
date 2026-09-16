//===- TaskEdgeContract.cpp ----------------------------------*- C++ -*-===//
//
// Implements the typed task-edge and payload contracts used by joint
// scheduling. This file deliberately has no scheduler integration.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {
namespace {

struct EdgeIdentity {
  uint32_t producer_task = 0;
  uint32_t consumer_task = 0;
  TaskEdgeKind kind = TaskEdgeKind::Control;
  TaskResultSegment producer_segment = TaskResultSegment::None;
  uint32_t producer_index = 0;
  TaskOperandSegment consumer_segment = TaskOperandSegment::Control;
  uint32_t consumer_index = 0;

  bool operator<(const EdgeIdentity &other) const {
    return std::tie(producer_task, consumer_task, kind, producer_segment,
                    producer_index, consumer_segment, consumer_index) <
           std::tie(other.producer_task, other.consumer_task, other.kind,
                    other.producer_segment, other.producer_index,
                    other.consumer_segment, other.consumer_index);
  }
};

struct ProducerResult {
  TaskflowTaskOp task;
  TaskResultSegment segment = TaskResultSegment::None;
  uint32_t index = 0;
  Value value;
};

std::optional<ProducerResult>
findProducerResult(Value value, const DenseMap<Operation *, unsigned> &task_ids,
                   std::string &error) {
  auto producer = value.getDefiningOp<TaskflowTaskOp>();
  if (!producer)
    return std::nullopt;
  if (!task_ids.count(producer.getOperation())) {
    error = "task edge producer is outside the selected function";
    return std::nullopt;
  }

  std::optional<ProducerResult> result;
  auto record = [&](TaskResultSegment segment, uint32_t index,
                    Value candidate) {
    if (candidate != value)
      return;
    if (result) {
      error = "task result appears in multiple result segments";
      return;
    }
    result = ProducerResult{producer, segment, index, candidate};
  };

  for (auto indexed : llvm::enumerate(producer.getDoneReads()))
    record(TaskResultSegment::DoneReads, indexed.index(), indexed.value());
  for (auto indexed : llvm::enumerate(producer.getDoneWrites()))
    record(TaskResultSegment::DoneWrites, indexed.index(), indexed.value());
  for (auto indexed : llvm::enumerate(producer.getValueOutputs()))
    record(TaskResultSegment::ValueOutputs, indexed.index(), indexed.value());
  if (!error.empty())
    return result;
  if (!result)
    error = "task edge operand references an unclassified task result";
  return result;
}

LogicalResult
collectProducerResults(Value value,
                       const DenseMap<Operation *, unsigned> &taskIds,
                       bool requireTaskProducer, bool crossedChannel,
                       SmallVectorImpl<ProducerResult> &results,
                       DenseSet<Value> &visited, std::string &error) {
  if (!visited.insert(value).second) {
    error = "task dependency forwarding graph contains a cycle or duplicate";
    return failure();
  }
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    if (channel.getSource().getType() != value.getType()) {
      error = "taskflow.channel source and result types do not match";
      return failure();
    }
    return collectProducerResults(channel.getSource(), taskIds,
                                  requireTaskProducer, true, results, visited,
                                  error);
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates()) {
      if (state.getType() != value.getType()) {
        error = "taskflow.join state and result types do not match";
        return failure();
      }
      if (failed(collectProducerResults(state, taskIds, requireTaskProducer,
                                        crossedChannel, results, visited,
                                        error)))
        return failure();
    }
    return success();
  }
  std::optional<ProducerResult> producer =
      findProducerResult(value, taskIds, error);
  if (!error.empty())
    return failure();
  if (producer) {
    results.push_back(*producer);
    return success();
  }
  if (crossedChannel && requireTaskProducer) {
    error = "explicit communication mode requires a channel to terminate in "
            "one or more task results";
    return failure();
  }
  return success();
}

std::optional<uint64_t> getScalarBitWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = integer.getWidth();
    if (width == 0)
      return std::nullopt;
    return width;
  }
  if (auto floating = dyn_cast<FloatType>(type)) {
    unsigned width = floating.getWidth();
    if (width == 0)
      return std::nullopt;
    return width;
  }
  if (auto complex = dyn_cast<ComplexType>(type)) {
    std::optional<uint64_t> element_width =
        getScalarBitWidth(complex.getElementType());
    if (!element_width || *element_width > UINT64_MAX / 2)
      return std::nullopt;
    return *element_width * 2;
  }
  return std::nullopt;
}

std::optional<uint64_t> getTypePayloadBits(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return getScalarBitWidth(type);
  if (!shaped.hasStaticShape())
    return std::nullopt;
  int64_t element_count = shaped.getNumElements();
  if (element_count < 0)
    return std::nullopt;
  std::optional<uint64_t> element_width =
      getTypePayloadBits(shaped.getElementType());
  if (!element_width || static_cast<uint64_t>(element_count) >
                            UINT64_MAX / std::max<uint64_t>(1, *element_width))
    return std::nullopt;
  return static_cast<uint64_t>(element_count) * *element_width;
}

std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

std::optional<uint64_t> getProducerPayloadBits(ProducerResult producer) {
  auto shaped = dyn_cast<ShapedType>(producer.value.getType());
  if (!shaped || producer.segment != TaskResultSegment::DoneWrites)
    return getTypePayloadBits(producer.value.getType());
  if (!shaped.hasStaticShape())
    return std::nullopt;

  SmallVector<TaskflowCounterOp> counters;
  for (Operation &operation : producer.task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
  if (counters.size() != static_cast<size_t>(shaped.getRank()))
    return getTypePayloadBits(producer.value.getType());

  uint64_t elements = 1;
  for (auto [axis, counter] : llvm::enumerate(counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *step != 1 || *lower < 0 ||
        *upper <= *lower || *upper > shaped.getShape()[axis])
      return std::nullopt;
    uint64_t extent = static_cast<uint64_t>(*upper - *lower);
    if (elements > UINT64_MAX / extent)
      return std::nullopt;
    elements *= extent;
  }
  std::optional<uint64_t> elementBits =
      getTypePayloadBits(shaped.getElementType());
  if (!elementBits || elements > UINT64_MAX / *elementBits)
    return std::nullopt;
  return elements * *elementBits;
}

std::optional<uint64_t> getExplicitPayloadBits(TaskflowTaskOp consumer,
                                               TaskOperandSegment segment,
                                               uint32_t index,
                                               std::string &error) {
  auto attribute = consumer->getAttr(kEdgePayloadBitsAttr);
  if (!attribute)
    return std::nullopt;
  auto payload_bits = dyn_cast<DenseI64ArrayAttr>(attribute);
  if (!payload_bits) {
    error = "amoeba.edge_payload_bits must be a DenseI64ArrayAttr";
    return std::nullopt;
  }

  size_t read_count = consumer.getWillReads().size();
  size_t value_count = consumer.getValueInputs().size();
  if (payload_bits.asArrayRef().size() != read_count + value_count) {
    error = "amoeba.edge_payload_bits must contain one entry for every "
            "will_reads and value_inputs operand";
    return std::nullopt;
  }
  if (segment != TaskOperandSegment::WillReads &&
      segment != TaskOperandSegment::ValueInputs) {
    error = "payload facts are only valid for data-carrying operands";
    return std::nullopt;
  }
  size_t payload_index = index;
  if (segment == TaskOperandSegment::ValueInputs)
    payload_index += read_count;
  int64_t bits = payload_bits[payload_index];
  if (bits < -1) {
    error = "amoeba.edge_payload_bits entries must be non-negative or -1";
    return std::nullopt;
  }
  if (bits == -1)
    return std::nullopt;
  return static_cast<uint64_t>(bits);
}

bool validateExplicitPayloadBits(TaskflowTaskOp consumer, std::string &error) {
  auto attribute = consumer->getAttr(kEdgePayloadBitsAttr);
  if (!attribute)
    return true;
  auto payload_bits = dyn_cast<DenseI64ArrayAttr>(attribute);
  if (!payload_bits) {
    error = "amoeba.edge_payload_bits must be a DenseI64ArrayAttr";
    return false;
  }
  size_t expected_size =
      consumer.getWillReads().size() + consumer.getValueInputs().size();
  if (payload_bits.asArrayRef().size() != expected_size) {
    error = "amoeba.edge_payload_bits must contain one entry for every "
            "will_reads and value_inputs operand";
    return false;
  }
  for (int64_t bits : payload_bits.asArrayRef()) {
    if (bits < -1) {
      error = "amoeba.edge_payload_bits entries must be non-negative or -1";
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<uint64_t> getStaticPayloadBits(Type type) {
  return getTypePayloadBits(type);
}

StringRef stringifyTaskEdgeKind(TaskEdgeKind kind) {
  switch (kind) {
  case TaskEdgeKind::Value:
    return "value";
  case TaskEdgeKind::Raw:
    return "raw";
  case TaskEdgeKind::War:
    return "war";
  case TaskEdgeKind::Waw:
    return "waw";
  case TaskEdgeKind::Control:
    return "control";
  }
  llvm_unreachable("unknown task edge kind");
}

StringRef stringifyTaskOperandSegment(TaskOperandSegment segment) {
  switch (segment) {
  case TaskOperandSegment::WillReads:
    return "will_reads";
  case TaskOperandSegment::WillWrites:
    return "will_writes";
  case TaskOperandSegment::ValueInputs:
    return "value_inputs";
  case TaskOperandSegment::Control:
    return "control";
  }
  llvm_unreachable("unknown task operand segment");
}

StringRef stringifyTaskResultSegment(TaskResultSegment segment) {
  switch (segment) {
  case TaskResultSegment::None:
    return "none";
  case TaskResultSegment::DoneReads:
    return "done_reads";
  case TaskResultSegment::DoneWrites:
    return "done_writes";
  case TaskResultSegment::ValueOutputs:
    return "value_outputs";
  }
  llvm_unreachable("unknown task result segment");
}

FailureOr<TaskEdgeGraph> buildTaskEdgeGraph(func::FuncOp func,
                                            const TaskEdgeGraphOptions &options,
                                            std::string &error) {
  error.clear();
  if (!func) {
    error = "cannot build task edges from a null function";
    return failure();
  }
  TaskEdgeGraph graph;
  func.walk([&](TaskflowTaskOp task) { graph.tasks_.push_back(task); });
  if (graph.tasks_.empty()) {
    error = "function contains no taskflow.task operations";
    return failure();
  }

  DenseMap<Operation *, unsigned> task_ids;
  llvm::StringMap<SmallVector<unsigned>> task_name_ids;
  for (auto indexed : llvm::enumerate(graph.tasks_)) {
    task_ids[indexed.value().getOperation()] = indexed.index();
    task_name_ids[indexed.value().getTaskName()].push_back(indexed.index());
  }

  std::set<EdgeIdentity> identities;
  auto addEdge = [&](ProducerResult producer, TaskflowTaskOp consumer,
                     TaskEdgeKind kind, TaskOperandSegment consumer_segment,
                     uint32_t consumer_index,
                     std::optional<uint64_t> payload) -> bool {
    unsigned producer_id = task_ids.lookup(producer.task.getOperation());
    unsigned consumer_id = task_ids.lookup(consumer.getOperation());
    if (producer_id == consumer_id) {
      error = "task edge cannot connect a task to itself";
      return false;
    }
    EdgeIdentity identity{producer_id,      consumer_id,    kind,
                          producer.segment, producer.index, consumer_segment,
                          consumer_index};
    if (!identities.insert(identity).second) {
      error = "duplicate task edge identity";
      return false;
    }
    if ((kind == TaskEdgeKind::Value || kind == TaskEdgeKind::Raw) &&
        !payload && options.require_payload) {
      error = "data-carrying task edge has unknown payload size";
      return false;
    }
    graph.edges_.push_back({producer.task, consumer, kind, producer.segment,
                            producer.index, consumer_segment, consumer_index,
                            payload});
    return true;
  };

  auto addOperandEdges = [&](TaskflowTaskOp consumer, ValueRange values,
                             TaskOperandSegment operand_segment,
                             TaskEdgeKind expected_kind) -> bool {
    for (auto indexed : llvm::enumerate(values)) {
      SmallVector<ProducerResult> producers;
      DenseSet<Value> visited;
      if (failed(collectProducerResults(
              indexed.value(), task_ids, options.require_payload,
              /*crossedChannel=*/false, producers, visited, error)))
        return false;
      if (producers.empty())
        continue;

      struct ResolvedEdge {
        ProducerResult producer;
        TaskEdgeKind kind;
        std::optional<uint64_t> payload;
      };
      SmallVector<ResolvedEdge> resolvedEdges;
      uint64_t inferredPayloadSum = 0;
      bool completePayload = true;
      bool hasDataEdge = false;
      for (ProducerResult producer : producers) {
        TaskEdgeKind kind = expected_kind;
        if (operand_segment == TaskOperandSegment::WillReads &&
            producer.segment != TaskResultSegment::DoneWrites) {
          error = "will_reads operand must resolve to done_writes results";
          return false;
        }
        if (operand_segment == TaskOperandSegment::WillWrites) {
          if (producer.segment == TaskResultSegment::DoneReads)
            kind = TaskEdgeKind::War;
          else if (producer.segment == TaskResultSegment::DoneWrites)
            kind = TaskEdgeKind::Waw;
          else {
            error = "will_writes operand must resolve to done_reads or "
                    "done_writes results";
            return false;
          }
        }
        if (operand_segment == TaskOperandSegment::ValueInputs &&
            producer.segment != TaskResultSegment::ValueOutputs) {
          error = "value_inputs operand must resolve to value_outputs results";
          return false;
        }
        std::optional<uint64_t> payload;
        if (kind == TaskEdgeKind::Value || kind == TaskEdgeKind::Raw) {
          hasDataEdge = true;
          payload = getProducerPayloadBits(producer);
          if (!payload)
            completePayload = false;
          else if (*payload > UINT64_MAX - inferredPayloadSum) {
            error = "joined task payload sum overflows unsigned 64-bit bits";
            return false;
          } else {
            inferredPayloadSum += *payload;
          }
        }
        resolvedEdges.push_back({producer, kind, payload});
      }

      std::optional<uint64_t> explicitPayload;
      if (hasDataEdge)
        explicitPayload = getExplicitPayloadBits(consumer, operand_segment,
                                                 indexed.index(), error);
      if (!error.empty())
        return false;
      if (explicitPayload) {
        if (resolvedEdges.size() == 1) {
          if (resolvedEdges.front().payload &&
              *resolvedEdges.front().payload != *explicitPayload) {
            error = "amoeba.edge_payload_bits disagrees with the static "
                    "payload type or transferred region";
            return false;
          }
          resolvedEdges.front().payload = explicitPayload;
        } else if (!completePayload || inferredPayloadSum != *explicitPayload) {
          error = "amoeba.edge_payload_bits cannot be partitioned across "
                  "taskflow.join producers";
          return false;
        }
      }
      for (const ResolvedEdge &resolved : resolvedEdges)
        if (!addEdge(resolved.producer, consumer, resolved.kind,
                     operand_segment, indexed.index(), resolved.payload))
          return false;
    }
    return true;
  };

  for (TaskflowTaskOp consumer : graph.tasks_) {
    if (!validateExplicitPayloadBits(consumer, error))
      return failure();
    if (!addOperandEdges(consumer, consumer.getWillReads(),
                         TaskOperandSegment::WillReads, TaskEdgeKind::Raw) ||
        !addOperandEdges(consumer, consumer.getWillWrites(),
                         TaskOperandSegment::WillWrites, TaskEdgeKind::Waw) ||
        !addOperandEdges(consumer, consumer.getValueInputs(),
                         TaskOperandSegment::ValueInputs, TaskEdgeKind::Value))
      return failure();

    if (!options.read_control_predecessors)
      continue;
    auto control_attr = consumer->getAttr(kControlPredecessorsAttr);
    if (!control_attr)
      continue;
    auto control_names = dyn_cast<ArrayAttr>(control_attr);
    if (!control_names) {
      error = "amoeba.control_predecessors must be an array attribute";
      return failure();
    }
    std::set<unsigned> control_producers;
    for (auto indexed : llvm::enumerate(control_names)) {
      StringRef producer_name;
      if (auto string = dyn_cast<StringAttr>(indexed.value()))
        producer_name = string.getValue();
      else if (auto symbol = dyn_cast<FlatSymbolRefAttr>(indexed.value()))
        producer_name = symbol.getValue();
      else {
        error = "amoeba.control_predecessors entries must be task-name "
                "strings or symbol references";
        return failure();
      }
      auto ids = task_name_ids.lookup(producer_name);
      if (ids.size() != 1) {
        error = ids.empty() ? "control predecessor task name does not exist: " +
                                  producer_name.str()
                            : "control predecessor task name is ambiguous: " +
                                  producer_name.str();
        return failure();
      }
      if (!control_producers.insert(ids.front()).second) {
        error =
            "duplicate control predecessor task name: " + producer_name.str();
        return failure();
      }
      TaskflowTaskOp producer = graph.tasks_[ids.front()];
      ProducerResult control_source{producer, TaskResultSegment::None, 0,
                                    Value()};
      if (!addEdge(control_source, consumer, TaskEdgeKind::Control,
                   TaskOperandSegment::Control, indexed.index(), std::nullopt))
        return failure();
    }
  }
  return graph;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
