//===- TaskCommunicationModel.h -----------------------------*- C++ -*-===//
//
// Opt-in communication model that adapts the explicit task-edge and
// architecture-level network contracts to TaskScheduler.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H
#define AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H

#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Adapts one strict typed task-edge graph and one validated directed network
// to the scheduler's coordinate-level transfer query.  A task-level edge
// does not identify a tile inside a multi-CGRA placement, so the scheduler
// uses the cheapest endpoint pair. Data payloads between the same task pair
// are added exactly, and each directed link is reserved until its transfer
// completes. Routing therefore minimizes load-aware arrival time.
class InterTaskNetworkCommunicationModel final
    : public ::mlir::taskflow::TaskCommunicationModel {
public:
  InterTaskNetworkCommunicationModel(TaskEdgeGraph edge_graph,
                                     InterTaskNetworkSpec network)
      : edge_graph_(std::move(edge_graph)), network_(std::move(network)),
        committedLinkIntervals_(network_.getLinks().size()),
        trialLinkIntervals_(network_.getLinks().size()),
        committedLocalIntervals_(network_.getRows() * network_.getColumns()),
        trialLocalIntervals_(network_.getRows() * network_.getColumns()) {}

  bool getTaskPredecessors(Operation *consumer,
                           llvm::SmallVectorImpl<Operation *> &predecessors,
                           std::string &error) const override;

  void resetReservations() override;
  void beginTrial() override;
  void finishTrial(bool commit) override;
  bool getTransferReadyCycle(Operation *producer, Operation *consumer,
                             int source_row, int source_col,
                             int destination_row, int destination_col,
                             int64_t producer_finish, bool reserve,
                             int64_t &ready_cycle, std::string &error) override;

  struct LinkInterval {
    uint32_t linkIndex = 0;
    int64_t startCycle = 0;
    int64_t endCycle = 0;
    bool localChannel = false;
    TaskNetworkCoordinate localCoordinate;
  };
  struct TransferRecord {
    Operation *producer = nullptr;
    Operation *consumer = nullptr;
    TaskNetworkCoordinate source;
    TaskNetworkCoordinate destination;
    uint64_t payloadBits = 0;
    uint64_t pathLatencyCycles = 0;
    uint64_t bottleneckBandwidthBitsPerCycle = 0;
    uint64_t transferCycles = 0;
    int64_t readyCycle = 0;
    llvm::SmallVector<uint32_t> edgeIndices;
    llvm::SmallVector<LinkInterval> links;
  };
  llvm::ArrayRef<TransferRecord> getCommittedTransfers() const {
    return committedTransfers_;
  }
  llvm::ArrayRef<TaskEdge> getTypedEdges() const {
    return edge_graph_.getEdges();
  }

private:
  bool routeTransfer(Operation *producer, Operation *consumer,
                     TaskNetworkCoordinate source,
                     TaskNetworkCoordinate destination, uint64_t payload_bits,
                     llvm::ArrayRef<uint32_t> edge_indices,
                     int64_t producer_finish, bool reserve,
                     int64_t &ready_cycle, std::string &error);

  TaskEdgeGraph edge_graph_;
  InterTaskNetworkSpec network_;
  using Interval = std::pair<int64_t, int64_t>;
  llvm::SmallVector<llvm::SmallVector<Interval>> committedLinkIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> trialLinkIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> committedLocalIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> trialLocalIntervals_;
  llvm::SmallVector<TransferRecord> committedTransfers_;
  llvm::SmallVector<TransferRecord> trialTransfers_;
};

// Builds the strict communication model for the current Neura architecture
// and checks that the network section dimensions match the scheduler grid.
FailureOr<std::unique_ptr<InterTaskNetworkCommunicationModel>>
createInterTaskNetworkCommunicationModel(func::FuncOp function, int grid_rows,
                                         int grid_columns, std::string &error);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H
