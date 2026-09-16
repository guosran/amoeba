// Orchestrates a materialized joint-scheduling candidate.

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

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

std::string hashBytes(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
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
    DenseSet<Value> visited;
    auto add_producer_from_value = [&](auto &&self, Value value) -> void {
      if (!visited.insert(value).second)
        return;
      if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
        self(self, channel.getSource());
        return;
      }
      if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
        for (Value state : join.getTileStates())
          self(self, state);
        return;
      }
      if (auto producer = value.getDefiningOp<TaskflowTaskOp>())
        addDependencyEdge(successors, producer.getOperation(), consumer_op);
    };

    for (Value value_input : consumer.getValueInputs()) {
      add_producer_from_value(add_producer_from_value, value_input);
    }
    for (Value operand : consumer->getOperands()) {
      add_producer_from_value(add_producer_from_value, operand);
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
  std::unique_ptr<
      amoeba::neura::joint_scheduling::InterTaskNetworkCommunicationModel>
      communication_model;
  if (communication_aware_) {
    std::string error;
    FailureOr<std::unique_ptr<
        amoeba::neura::joint_scheduling::InterTaskNetworkCommunicationModel>>
        created = amoeba::neura::joint_scheduling::
            createInterTaskNetworkCommunicationModel(func, grid_rows_,
                                                     grid_cols_, error);
    if (failed(created)) {
      func.emitError() << error;
      return false;
    }
    communication_model = std::move(*created);
  }
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_,
                          ShapeSelectionPolicy::FixedOrientation,
                          communication_model.get());
  if (!scheduler.schedule(func, priority)) {
    return false;
  }
  bool mapperReplayVerified = true;
  const bool verifyMapperReplay =
      func->hasAttr("joint_scheduling_mapper_replay_completed");
  if (verifyMapperReplay)
    func.walk([&](TaskflowTaskOp task) {
      if (!task->hasAttr("amoeba.mapper_replay_verified")) {
        task.emitError()
            << "joint replay requires a real per-task mapper result";
        mapperReplayVerified = false;
        return;
      }
      auto selectedCount =
          task->getAttrOfType<IntegerAttr>("amoeba.selected_cgra_count");
      auto selectedShape =
          task->getAttrOfType<StringAttr>("amoeba.selected_cgra_shape");
      auto info =
          task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
      auto positions =
          info ? info.getAs<ArrayAttr>("cgra_positions") : ArrayAttr();
      if (!selectedCount || !selectedShape || !positions ||
          selectedCount.getInt() != static_cast<int64_t>(positions.size())) {
        task.emitError() << "mapped shape and scheduled placement disagree";
        mapperReplayVerified = false;
        return;
      }
      int64_t minRow = std::numeric_limits<int64_t>::max();
      int64_t maxRow = -1;
      int64_t minCol = std::numeric_limits<int64_t>::max();
      int64_t maxCol = -1;
      for (Attribute attribute : positions) {
        auto position = dyn_cast<DictionaryAttr>(attribute);
        auto row =
            position ? position.getAs<IntegerAttr>("row") : IntegerAttr();
        auto col =
            position ? position.getAs<IntegerAttr>("col") : IntegerAttr();
        if (!row || !col) {
          mapperReplayVerified = false;
          return;
        }
        minRow = std::min(minRow, row.getInt());
        maxRow = std::max(maxRow, row.getInt());
        minCol = std::min(minCol, col.getInt());
        maxCol = std::max(maxCol, col.getInt());
      }
      std::string actual = std::to_string(maxRow - minRow + 1) + "x" +
                           std::to_string(maxCol - minCol + 1);
      if (actual != selectedShape.getValue()) {
        task.emitError() << "scheduled rectangle does not preserve selected "
                            "shape orientation";
        mapperReplayVerified = false;
      }
    });
  if (verifyMapperReplay && !mapperReplayVerified)
    return false;
  if (!verifyMapperReplay)
    func->setAttr("joint_scheduling_predicted_makespan",
                  IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                   scheduler.getScheduleMakespan()));
  if (verifyMapperReplay) {
    func->setAttr("joint_scheduling_actual_makespan",
                  IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                   scheduler.getScheduleMakespan()));
    func->setAttr("joint_scheduling_replay_verified",
                  UnitAttr::get(func.getContext()));
  }
  if (communication_model) {
    OpBuilder builder(func.getContext());
    SmallVector<Attribute> transfers;
    for (const auto &record : communication_model->getCommittedTransfers()) {
      if (record.payloadBits >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          record.pathLatencyCycles >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          record.bottleneckBandwidthBitsPerCycle >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          record.transferCycles >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        func.emitError() << "communication trace value exceeds signed 64-bit";
        return false;
      }
      SmallVector<NamedAttribute> transfer;
      auto producer = cast<TaskflowTaskOp>(record.producer);
      auto consumer = cast<TaskflowTaskOp>(record.consumer);
      transfer.push_back(builder.getNamedAttr(
          "producer", builder.getStringAttr(producer.getTaskName())));
      transfer.push_back(builder.getNamedAttr(
          "consumer", builder.getStringAttr(consumer.getTaskName())));
      transfer.push_back(builder.getNamedAttr(
          "payload_bits", builder.getI64IntegerAttr(record.payloadBits)));
      transfer.push_back(builder.getNamedAttr(
          "path_latency_cycles",
          builder.getI64IntegerAttr(record.pathLatencyCycles)));
      transfer.push_back(builder.getNamedAttr(
          "bottleneck_bandwidth_bits_per_cycle",
          builder.getI64IntegerAttr(record.bottleneckBandwidthBitsPerCycle)));
      transfer.push_back(builder.getNamedAttr(
          "transfer_cycles", builder.getI64IntegerAttr(record.transferCycles)));
      transfer.push_back(builder.getNamedAttr(
          "ready_cycle", builder.getI64IntegerAttr(record.readyCycle)));
      transfer.push_back(builder.getNamedAttr(
          "source_col", builder.getI32IntegerAttr(record.source.column)));
      transfer.push_back(builder.getNamedAttr(
          "source_row", builder.getI32IntegerAttr(record.source.row)));
      transfer.push_back(builder.getNamedAttr(
          "destination_col",
          builder.getI32IntegerAttr(record.destination.column)));
      transfer.push_back(builder.getNamedAttr(
          "destination_row",
          builder.getI32IntegerAttr(record.destination.row)));
      SmallVector<Attribute> edgeIndices;
      for (uint32_t edgeIndex : record.edgeIndices)
        edgeIndices.push_back(builder.getI32IntegerAttr(edgeIndex));
      transfer.push_back(builder.getNamedAttr(
          "edge_indices", builder.getArrayAttr(edgeIndices)));
      SmallVector<Attribute> links;
      for (const auto &interval : record.links) {
        SmallVector<NamedAttribute> link;
        link.push_back(builder.getNamedAttr(
            "end_cycle", builder.getI64IntegerAttr(interval.endCycle)));
        if (interval.localChannel) {
          link.push_back(builder.getNamedAttr(
              "col",
              builder.getI32IntegerAttr(interval.localCoordinate.column)));
          link.push_back(builder.getNamedAttr(
              "resource_kind", builder.getStringAttr("local_channel")));
          link.push_back(builder.getNamedAttr(
              "row", builder.getI32IntegerAttr(interval.localCoordinate.row)));
        } else {
          link.push_back(builder.getNamedAttr(
              "link_index", builder.getI32IntegerAttr(interval.linkIndex)));
          link.push_back(builder.getNamedAttr(
              "resource_kind", builder.getStringAttr("network_link")));
        }
        link.push_back(builder.getNamedAttr(
            "start_cycle", builder.getI64IntegerAttr(interval.startCycle)));
        links.push_back(builder.getDictionaryAttr(link));
      }
      transfer.push_back(
          builder.getNamedAttr("links", builder.getArrayAttr(links)));
      transfers.push_back(builder.getDictionaryAttr(transfer));
    }
    func->setAttr("joint_scheduling_communication_trace",
                  builder.getArrayAttr(transfers));
    SmallVector<Attribute> dependencies;
    for (const auto &edge : communication_model->getTypedEdges()) {
      taskflow::TaskflowTaskOp producer = edge.producer;
      taskflow::TaskflowTaskOp consumer = edge.consumer;
      SmallVector<NamedAttribute> dependency;
      dependency.push_back(builder.getNamedAttr(
          "consumer", builder.getStringAttr(consumer.getTaskName())));
      dependency.push_back(builder.getNamedAttr(
          "consumer_index", builder.getI32IntegerAttr(edge.consumer_index)));
      dependency.push_back(builder.getNamedAttr(
          "consumer_segment",
          builder.getStringAttr(
              amoeba::neura::joint_scheduling::stringifyTaskOperandSegment(
                  edge.consumer_segment))));
      dependency.push_back(builder.getNamedAttr(
          "kind", builder.getStringAttr(
                      amoeba::neura::joint_scheduling::stringifyTaskEdgeKind(
                          edge.kind))));
      if (edge.payload_bits)
        dependency.push_back(builder.getNamedAttr(
            "payload_bits", builder.getI64IntegerAttr(*edge.payload_bits)));
      dependency.push_back(builder.getNamedAttr(
          "producer", builder.getStringAttr(producer.getTaskName())));
      dependency.push_back(builder.getNamedAttr(
          "producer_index", builder.getI32IntegerAttr(edge.producer_index)));
      dependency.push_back(builder.getNamedAttr(
          "producer_segment",
          builder.getStringAttr(
              amoeba::neura::joint_scheduling::stringifyTaskResultSegment(
                  edge.producer_segment))));
      dependencies.push_back(builder.getDictionaryAttr(dependency));
    }
    func->setAttr("joint_scheduling_dependency_trace",
                  builder.getArrayAttr(dependencies));
  }
  if (verifyMapperReplay) {
    auto candidateId =
        func->getAttrOfType<StringAttr>("joint_scheduling_candidate_id");
    auto architectureSha =
        func->getAttrOfType<StringAttr>("joint_scheduling_architecture_sha256");
    if (!candidateId || !architectureSha) {
      func.emitError() << "mapper replay lacks its candidate or architecture "
                          "trace identity";
      return false;
    }

    OpBuilder builder(func.getContext());
    SmallVector<Attribute> taskSchedule;
    SmallVector<Attribute> dependencies;
    SmallVector<Attribute> routes;
    std::string traceIdentity;
    llvm::raw_string_ostream identity(traceIdentity);
    StringRef communicationMode = communication_aware_ ? "explicit" : "none";
    identity << "joint-schedule-trace-v2\n"
             << candidateId.getValue() << "\n"
             << architectureSha.getValue() << "\n"
             << communicationMode << "\n";

    bool completeTrace = true;
    func.walk([&](TaskflowTaskOp task) {
      if (!completeTrace)
        return;
      auto scheduleEntry =
          llvm::find_if(scheduler.getScheduleEntries(), [&](const auto &entry) {
            return entry.task == task.getOperation();
          });
      if (scheduleEntry == scheduler.getScheduleEntries().end()) {
        task.emitError() << "scheduler omitted task from actual replay trace";
        completeTrace = false;
        return;
      }
      SmallVector<NamedAttribute> taskRecord;
      taskRecord.push_back(builder.getNamedAttr(
          "task", builder.getStringAttr(task.getTaskName())));
      taskRecord.push_back(builder.getNamedAttr(
          "start_cycle", builder.getI64IntegerAttr(scheduleEntry->startCycle)));
      taskRecord.push_back(builder.getNamedAttr(
          "end_cycle", builder.getI64IntegerAttr(scheduleEntry->endCycle)));
      identity << "T|" << task.getTaskName().size() << ":" << task.getTaskName()
               << "|" << scheduleEntry->startCycle << "|"
               << scheduleEntry->endCycle;
      SmallVector<Attribute> cells;
      for (auto [row, col] : scheduleEntry->positions) {
        cells.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("col", builder.getI32IntegerAttr(col)),
            builder.getNamedAttr("row", builder.getI32IntegerAttr(row)),
        }));
        identity << "|" << row << "," << col;
      }
      identity << "\n";
      taskRecord.push_back(
          builder.getNamedAttr("cgra_positions", builder.getArrayAttr(cells)));
      taskSchedule.push_back(builder.getDictionaryAttr(taskRecord));
    });
    if (!completeTrace)
      return false;

    if (communication_model) {
      for (const auto &edge : communication_model->getTypedEdges()) {
        TaskflowTaskOp producer = edge.producer;
        TaskflowTaskOp consumer = edge.consumer;
        SmallVector<NamedAttribute> dependency;
        dependency.push_back(builder.getNamedAttr(
            "consumer", builder.getStringAttr(consumer.getTaskName())));
        dependency.push_back(builder.getNamedAttr(
            "consumer_index", builder.getI32IntegerAttr(edge.consumer_index)));
        dependency.push_back(builder.getNamedAttr(
            "consumer_segment",
            builder.getStringAttr(
                amoeba::neura::joint_scheduling::stringifyTaskOperandSegment(
                    edge.consumer_segment))));
        dependency.push_back(builder.getNamedAttr(
            "kind", builder.getStringAttr(
                        amoeba::neura::joint_scheduling::stringifyTaskEdgeKind(
                            edge.kind))));
        if (edge.payload_bits)
          dependency.push_back(builder.getNamedAttr(
              "payload_bits", builder.getI64IntegerAttr(*edge.payload_bits)));
        dependency.push_back(builder.getNamedAttr(
            "producer", builder.getStringAttr(producer.getTaskName())));
        dependency.push_back(builder.getNamedAttr(
            "producer_index", builder.getI32IntegerAttr(edge.producer_index)));
        dependency.push_back(builder.getNamedAttr(
            "producer_segment",
            builder.getStringAttr(
                amoeba::neura::joint_scheduling::stringifyTaskResultSegment(
                    edge.producer_segment))));
        dependencies.push_back(builder.getDictionaryAttr(dependency));
        identity
            << "D|" << producer.getTaskName().size() << ":"
            << producer.getTaskName() << "|" << consumer.getTaskName().size()
            << ":" << consumer.getTaskName() << "|"
            << amoeba::neura::joint_scheduling::stringifyTaskEdgeKind(edge.kind)
            << "|"
            << amoeba::neura::joint_scheduling::stringifyTaskResultSegment(
                   edge.producer_segment)
            << "|" << edge.producer_index << "|"
            << amoeba::neura::joint_scheduling::stringifyTaskOperandSegment(
                   edge.consumer_segment)
            << "|" << edge.consumer_index << "|";
        if (edge.payload_bits)
          identity << *edge.payload_bits;
        else
          identity << "-";
        identity << "\n";
      }
      for (const auto &record : communication_model->getCommittedTransfers()) {
        if (record.payloadBits >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            record.pathLatencyCycles >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            record.bottleneckBandwidthBitsPerCycle >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            record.transferCycles >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          func.emitError() << "actual communication trace value exceeds "
                              "signed 64-bit";
          return false;
        }
        TaskflowTaskOp producer = cast<TaskflowTaskOp>(record.producer);
        TaskflowTaskOp consumer = cast<TaskflowTaskOp>(record.consumer);
        SmallVector<NamedAttribute> route;
        route.push_back(builder.getNamedAttr(
            "consumer", builder.getStringAttr(consumer.getTaskName())));
        route.push_back(builder.getNamedAttr(
            "destination_col",
            builder.getI32IntegerAttr(record.destination.column)));
        route.push_back(builder.getNamedAttr(
            "destination_row",
            builder.getI32IntegerAttr(record.destination.row)));
        SmallVector<Attribute> edgeIndices;
        for (uint32_t edgeIndex : record.edgeIndices)
          edgeIndices.push_back(builder.getI32IntegerAttr(edgeIndex));
        route.push_back(builder.getNamedAttr(
            "edge_indices", builder.getArrayAttr(edgeIndices)));
        route.push_back(builder.getNamedAttr(
            "payload_bits", builder.getI64IntegerAttr(record.payloadBits)));
        route.push_back(builder.getNamedAttr(
            "path_latency_cycles",
            builder.getI64IntegerAttr(record.pathLatencyCycles)));
        route.push_back(builder.getNamedAttr(
            "bottleneck_bandwidth_bits_per_cycle",
            builder.getI64IntegerAttr(record.bottleneckBandwidthBitsPerCycle)));
        route.push_back(builder.getNamedAttr(
            "transfer_cycles",
            builder.getI64IntegerAttr(record.transferCycles)));
        route.push_back(builder.getNamedAttr(
            "producer", builder.getStringAttr(producer.getTaskName())));
        route.push_back(builder.getNamedAttr(
            "ready_cycle", builder.getI64IntegerAttr(record.readyCycle)));
        route.push_back(builder.getNamedAttr(
            "source_col", builder.getI32IntegerAttr(record.source.column)));
        route.push_back(builder.getNamedAttr(
            "source_row", builder.getI32IntegerAttr(record.source.row)));
        identity << "R|" << producer.getTaskName().size() << ":"
                 << producer.getTaskName() << "|"
                 << consumer.getTaskName().size() << ":"
                 << consumer.getTaskName() << "|" << record.source.row << ","
                 << record.source.column << "|" << record.destination.row << ","
                 << record.destination.column << "|" << record.payloadBits
                 << "|" << record.readyCycle << "|" << record.pathLatencyCycles
                 << "|" << record.bottleneckBandwidthBitsPerCycle << "|"
                 << record.transferCycles;
        for (uint32_t edgeIndex : record.edgeIndices)
          identity << "|e" << edgeIndex;
        SmallVector<Attribute> linkIntervals;
        for (const auto &interval : record.links) {
          SmallVector<NamedAttribute> resource;
          resource.push_back(builder.getNamedAttr(
              "end_cycle", builder.getI64IntegerAttr(interval.endCycle)));
          if (interval.localChannel) {
            resource.push_back(builder.getNamedAttr(
                "col",
                builder.getI32IntegerAttr(interval.localCoordinate.column)));
            resource.push_back(builder.getNamedAttr(
                "resource_kind", builder.getStringAttr("local_channel")));
            resource.push_back(builder.getNamedAttr(
                "row",
                builder.getI32IntegerAttr(interval.localCoordinate.row)));
            identity << "|c" << interval.localCoordinate.row << ","
                     << interval.localCoordinate.column << ","
                     << interval.startCycle << "," << interval.endCycle;
          } else {
            resource.push_back(builder.getNamedAttr(
                "link_index", builder.getI32IntegerAttr(interval.linkIndex)));
            resource.push_back(builder.getNamedAttr(
                "resource_kind", builder.getStringAttr("network_link")));
            identity << "|l" << interval.linkIndex << "," << interval.startCycle
                     << "," << interval.endCycle;
          }
          resource.push_back(builder.getNamedAttr(
              "start_cycle", builder.getI64IntegerAttr(interval.startCycle)));
          linkIntervals.push_back(builder.getDictionaryAttr(resource));
        }
        identity << "\n";
        route.push_back(
            builder.getNamedAttr("links", builder.getArrayAttr(linkIntervals)));
        routes.push_back(builder.getDictionaryAttr(route));
      }
    }
    identity.flush();
    func->setAttr("joint_scheduling_actual_trace_sha256",
                  builder.getStringAttr(hashBytes(traceIdentity)));
    func->setAttr(
        "joint_scheduling_actual_trace",
        builder.getDictionaryAttr({
            builder.getNamedAttr("architecture_sha256", architectureSha),
            builder.getNamedAttr("candidate_id", candidateId),
            builder.getNamedAttr("communication_mode",
                                 builder.getStringAttr(communicationMode)),
            builder.getNamedAttr("dependencies",
                                 builder.getArrayAttr(dependencies)),
            builder.getNamedAttr("routes", builder.getArrayAttr(routes)),
            builder.getNamedAttr("task_schedule",
                                 builder.getArrayAttr(taskSchedule)),
        }));
  }
  func.walk([](TaskflowTaskOp task) {
    task->removeAttr("amoeba.joint_shape_orientation_fixed");
  });
  return true;
}

} // namespace taskflow
} // namespace mlir
