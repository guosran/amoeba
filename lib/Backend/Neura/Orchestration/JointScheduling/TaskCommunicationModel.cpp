//===- TaskCommunicationModel.cpp ---------------------------*- C++ -*-===//
//
// Adapts the explicit task-edge and inter-task network contracts to the
// optional communication-aware TaskScheduler mode.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "Backend/Neura/NeuraBackendOptions.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

bool InterTaskNetworkCommunicationModel::getTaskPredecessors(
    Operation *consumer, llvm::SmallVectorImpl<Operation *> &predecessors,
    std::string &error) const {
  error.clear();
  predecessors.clear();
  if (!consumer) {
    error = "communication predecessor query has a null consumer";
    return false;
  }

  llvm::DenseSet<Operation *> seen;
  for (const TaskEdge &edge : edge_graph_.getEdges()) {
    if (edge.consumer != consumer)
      continue;
    taskflow::TaskflowTaskOp producer_task = edge.producer;
    Operation *producer = producer_task.getOperation();
    if (seen.insert(producer).second)
      predecessors.push_back(producer);
  }
  return true;
}

void InterTaskNetworkCommunicationModel::resetReservations() {
  for (auto &intervals : committedLinkIntervals_)
    intervals.clear();
  for (auto &intervals : committedLocalIntervals_)
    intervals.clear();
  trialLinkIntervals_ = committedLinkIntervals_;
  trialLocalIntervals_ = committedLocalIntervals_;
  committedTransfers_.clear();
  trialTransfers_.clear();
}

void InterTaskNetworkCommunicationModel::beginTrial() {
  trialLinkIntervals_ = committedLinkIntervals_;
  trialLocalIntervals_ = committedLocalIntervals_;
  trialTransfers_.clear();
}

void InterTaskNetworkCommunicationModel::finishTrial(bool commit) {
  if (commit) {
    committedLinkIntervals_ = trialLinkIntervals_;
    committedLocalIntervals_ = trialLocalIntervals_;
    committedTransfers_.append(trialTransfers_.begin(), trialTransfers_.end());
  }
  trialTransfers_.clear();
}

static bool checkedAddCycle(int64_t lhs, uint64_t rhs, int64_t &result) {
  if (lhs < 0 ||
      rhs > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      lhs > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(rhs))
    return false;
  result = lhs + static_cast<int64_t>(rhs);
  return true;
}

bool InterTaskNetworkCommunicationModel::routeTransfer(
    Operation *producer, Operation *consumer, TaskNetworkCoordinate source,
    TaskNetworkCoordinate destination, uint64_t payload_bits,
    ArrayRef<uint32_t> edge_indices, int64_t producer_finish, bool reserve,
    int64_t &ready_cycle, std::string &error) {
  const int rows = network_.getRows();
  const int columns = network_.getColumns();
  auto inBounds = [&](TaskNetworkCoordinate coordinate) {
    return coordinate.row >= 0 && coordinate.row < rows &&
           coordinate.column >= 0 && coordinate.column < columns;
  };
  if (!inBounds(source) || !inBounds(destination) || producer_finish < 0) {
    error = "communication transfer endpoint or start cycle is invalid";
    return false;
  }

  TransferRecord record;
  record.producer = producer;
  record.consumer = consumer;
  record.source = source;
  record.destination = destination;
  record.payloadBits = payload_bits;
  record.edgeIndices.assign(edge_indices.begin(), edge_indices.end());
  if (source == destination) {
    const uint64_t bandwidth = network_.getLocalBandwidthBitsPerCycle();
    if (bandwidth == 0) {
      error = "local communication bandwidth must be positive";
      return false;
    }
    const uint64_t cycles =
        payload_bits == 0 ? 0 : ((payload_bits - 1) / bandwidth) + 1;
    record.pathLatencyCycles = 0;
    record.bottleneckBandwidthBitsPerCycle = bandwidth;
    record.transferCycles = cycles;
    const int local_index = source.row * columns + source.column;
    int64_t start = producer_finish;
    while (true) {
      int64_t finish = 0;
      if (!checkedAddCycle(start, cycles, finish)) {
        error = "local communication ready cycle overflows signed 64-bit";
        return false;
      }
      int64_t next_start = start;
      for (auto [occupied_start, occupied_end] :
           trialLocalIntervals_[local_index]) {
        if (start < occupied_end && finish > occupied_start)
          next_start = std::max(next_start, occupied_end);
      }
      if (next_start == start) {
        ready_cycle = finish;
        break;
      }
      start = next_start;
    }
    record.links.push_back({0, start, ready_cycle, true, source});
    record.readyCycle = ready_cycle;
    if (reserve) {
      auto &intervals = trialLocalIntervals_[local_index];
      intervals.push_back({start, ready_cycle});
      llvm::sort(intervals);
      trialTransfers_.push_back(std::move(record));
    }
    return true;
  }

  const int node_count = rows * columns;
  const int source_id = source.row * columns + source.column;
  const int destination_id = destination.row * columns + destination.column;
  ArrayRef<InterTaskNetworkLink> links = network_.getLinks();
  struct RouteChoice {
    SmallVector<uint32_t> links;
    int64_t start = 0;
    int64_t finish = std::numeric_limits<int64_t>::max();
    uint64_t latency = 0;
    uint64_t bottleneck = 0;
    uint64_t duration = 0;
  };
  RouteChoice best;
  SmallVector<uint32_t> path;
  std::vector<uint8_t> visited(node_count, 0);
  visited[source_id] = 1;
  bool arithmetic_error = false;
  bool search_budget_exceeded = false;
  uint64_t search_steps = 0;
  constexpr uint64_t kRouteSearchStepBudget = 100000;

  auto evaluate = [&]() {
    uint64_t latency = 0;
    uint64_t bottleneck = UINT64_MAX;
    for (uint32_t link_index : path) {
      const InterTaskNetworkLink &link = links[link_index];
      if (link.bandwidth_bits_per_cycle == 0 ||
          latency > UINT64_MAX - link.latency_cycles) {
        arithmetic_error = true;
        return;
      }
      latency += link.latency_cycles;
      bottleneck = std::min(bottleneck, link.bandwidth_bits_per_cycle);
    }
    const uint64_t transmission =
        payload_bits == 0 ? 0 : ((payload_bits - 1) / bottleneck) + 1;
    if (latency > UINT64_MAX - transmission) {
      arithmetic_error = true;
      return;
    }
    const uint64_t duration = latency + transmission;
    int64_t start = producer_finish;
    int64_t finish = 0;
    // Finds the first common half-open gap across all route links. This stays
    // exact even when transfers are committed out of start-time order.
    while (true) {
      if (!checkedAddCycle(start, duration, finish)) {
        arithmetic_error = true;
        return;
      }
      int64_t next_start = start;
      for (uint32_t link_index : path) {
        for (auto [occupied_start, occupied_end] :
             trialLinkIntervals_[link_index]) {
          if (start < occupied_end && finish > occupied_start)
            next_start = std::max(next_start, occupied_end);
        }
      }
      if (next_start == start)
        break;
      start = next_start;
    }
    if (finish < best.finish ||
        (finish == best.finish &&
         std::lexicographical_compare(path.begin(), path.end(),
                                      best.links.begin(), best.links.end())))
      best = {path, start, finish, latency, bottleneck, duration};
  };

  std::function<void(int)> enumerate = [&](int current) {
    if (search_budget_exceeded)
      return;
    if (++search_steps > kRouteSearchStepBudget) {
      search_budget_exceeded = true;
      return;
    }
    if (current == destination_id) {
      evaluate();
      return;
    }
    for (auto indexed : llvm::enumerate(links)) {
      const InterTaskNetworkLink &link = indexed.value();
      const int link_source = link.source.row * columns + link.source.column;
      const int next = link.destination.row * columns + link.destination.column;
      if (link_source != current || visited[next])
        continue;
      visited[next] = 1;
      path.push_back(static_cast<uint32_t>(indexed.index()));
      enumerate(next);
      path.pop_back();
      visited[next] = 0;
    }
  };
  enumerate(source_id);
  if (search_budget_exceeded) {
    error = "communication route search exceeds its deterministic budget";
    return false;
  }
  if (arithmetic_error) {
    error = "communication route cost overflows its 64-bit contract";
    return false;
  }
  if (best.links.empty()) {
    error = "communication network has no directed path between endpoints";
    return false;
  }

  for (uint32_t link_index : best.links) {
    record.links.push_back({link_index, best.start, best.finish, false, {}});
    if (reserve) {
      auto &intervals = trialLinkIntervals_[link_index];
      intervals.push_back({best.start, best.finish});
      llvm::sort(intervals);
    }
  }
  ready_cycle = best.finish;
  record.pathLatencyCycles = best.latency;
  record.bottleneckBandwidthBitsPerCycle = best.bottleneck;
  record.transferCycles = best.duration;
  record.readyCycle = best.finish;
  if (reserve)
    trialTransfers_.push_back(std::move(record));
  return true;
}

bool InterTaskNetworkCommunicationModel::getTransferReadyCycle(
    Operation *producer, Operation *consumer, int source_row, int source_col,
    int destination_row, int destination_col, int64_t producer_finish,
    bool reserve, int64_t &ready_cycle, std::string &error) {
  error.clear();
  ready_cycle = producer_finish;
  if (!producer || !consumer) {
    error = "communication transfer query has a null task endpoint";
    return false;
  }
  if (source_row < 0 || source_col < 0 || destination_row < 0 ||
      destination_col < 0) {
    error = "communication transfer endpoint is outside the grid";
    return false;
  }

  bool found_edge = false;
  uint64_t payload_bits = 0;
  SmallVector<uint32_t> edge_indices;
  for (auto indexed : llvm::enumerate(edge_graph_.getEdges())) {
    const TaskEdge &edge = indexed.value();
    taskflow::TaskflowTaskOp producer_task = edge.producer;
    taskflow::TaskflowTaskOp consumer_task = edge.consumer;
    if (producer_task.getOperation() != producer ||
        consumer_task.getOperation() != consumer)
      continue;
    found_edge = true;
    if (edge.kind != TaskEdgeKind::Value && edge.kind != TaskEdgeKind::Raw)
      continue;
    edge_indices.push_back(static_cast<uint32_t>(indexed.index()));
    if (!edge.payload_bits) {
      error = "communication transfer has an unknown Value/RAW payload";
      return false;
    }
    if (*edge.payload_bits > UINT64_MAX - payload_bits) {
      error = "communication payload sum overflows unsigned 64-bit bits";
      return false;
    }
    payload_bits += *edge.payload_bits;
  }

  if (!found_edge) {
    error = "communication transfer query does not match a typed task edge";
    return false;
  }
  if (payload_bits == 0)
    return true;
  return routeTransfer(producer, consumer, {source_row, source_col},
                       {destination_row, destination_col}, payload_bits,
                       edge_indices, producer_finish, reserve, ready_cycle,
                       error);
}

FailureOr<std::unique_ptr<InterTaskNetworkCommunicationModel>>
createInterTaskNetworkCommunicationModel(func::FuncOp function, int grid_rows,
                                         int grid_columns, std::string &error) {
  error.clear();
  TaskEdgeGraphOptions edge_options;
  edge_options.require_payload = true;
  edge_options.read_control_predecessors = true;
  FailureOr<TaskEdgeGraph> edge_graph =
      buildTaskEdgeGraph(function, edge_options, error);
  if (failed(edge_graph))
    return failure();

  FailureOr<std::optional<InterTaskNetworkSpec>> network =
      loadInterTaskNetworkSpec(::mlir::amoeba::getNeuraArchitectureSpecFile(),
                               true, error);
  if (failed(network) || !network->has_value()) {
    if (succeeded(network) && !network->has_value())
      error = "communication model requires an inter_task_network section";
    return failure();
  }
  if (!network->value().validateForGrid(grid_rows, grid_columns, error))
    return failure();

  return std::make_unique<InterTaskNetworkCommunicationModel>(
      std::move(*edge_graph), std::move(**network));
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
