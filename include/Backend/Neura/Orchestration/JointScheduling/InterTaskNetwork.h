//===- InterTaskNetwork.h ------------------------------------*- C++ -*-===//
//
// Public architecture-level inter-task network contract for joint scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_INTER_TASK_NETWORK_H
#define AMOEBA_JOINT_SCHEDULING_INTER_TASK_NETWORK_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// A coordinate in the multi-CGRA task-level fabric.  These are not tile
// coordinates inside Neura's per-CGRA architecture.
struct TaskNetworkCoordinate {
  int row = 0;
  int column = 0;

  bool operator==(const TaskNetworkCoordinate &other) const {
    return row == other.row && column == other.column;
  }
};

// One directed inter-task link.  Every link carries its own positive values;
// no value is inherited from Neura's intra-CGRA link_defaults.
struct InterTaskNetworkLink {
  TaskNetworkCoordinate source;
  TaskNetworkCoordinate destination;
  uint64_t latency_cycles = 0;
  uint64_t bandwidth_bits_per_cycle = 0;
};

struct InterTaskNetworkPath {
  TaskNetworkCoordinate source;
  TaskNetworkCoordinate destination;
  llvm::SmallVector<InterTaskNetworkLink> links;
  uint64_t latency_cycles = 0;
  uint64_t bottleneck_bandwidth_bits_per_cycle = 0;
};

// Versioned architecture-level task network.  The v1 schema uses an explicit
// directed link list and deterministic shortest-hop routing with
// row/column-lexicographic tie breaking.
class InterTaskNetworkSpec {
public:
  static constexpr uint64_t kSchemaVersion = 1;

  InterTaskNetworkSpec() = default;
  InterTaskNetworkSpec(uint64_t version, int rows, int columns,
                       uint64_t local_bandwidth_bits_per_cycle,
                       llvm::SmallVector<InterTaskNetworkLink> links)
      : version_(version), rows_(rows), columns_(columns),
        local_bandwidth_bits_per_cycle_(local_bandwidth_bits_per_cycle),
        links_(std::move(links)) {}

  uint64_t getVersion() const { return version_; }
  int getRows() const { return rows_; }
  int getColumns() const { return columns_; }
  uint64_t getLocalBandwidthBitsPerCycle() const {
    return local_bandwidth_bits_per_cycle_;
  }
  llvm::ArrayRef<InterTaskNetworkLink> getLinks() const {
    return {links_.data(), links_.size()};
  }

  // Validates the network itself and its dimensions against the scheduler's
  // multi-CGRA grid.
  bool validateForGrid(int grid_rows, int grid_columns,
                       std::string &error) const;

  // Finds a directed path according to the v1 deterministic route policy.
  FailureOr<InterTaskNetworkPath> route(TaskNetworkCoordinate source,
                                        TaskNetworkCoordinate destination,
                                        std::string &error) const;

  // Computes path latency plus ceil(payload_bits / bottleneck bandwidth),
  // checked against signed 64-bit scheduler cycles.  Same-coordinate traffic
  // uses the explicit local bandwidth and zero path latency.
  FailureOr<int64_t> transferCycles(TaskNetworkCoordinate source,
                                    TaskNetworkCoordinate destination,
                                    uint64_t payload_bits,
                                    std::string &error) const;

private:
  uint64_t version_ = 0;
  int rows_ = 0;
  int columns_ = 0;
  uint64_t local_bandwidth_bits_per_cycle_ = 0;
  llvm::SmallVector<InterTaskNetworkLink> links_;
};

// Parses an architecture YAML document containing an optional
// `inter_task_network` mapping.  If require_network is false, a document with
// no such mapping returns std::nullopt.  If it is true, absence is an error.
FailureOr<std::optional<InterTaskNetworkSpec>>
parseInterTaskNetworkYaml(llvm::StringRef yaml, bool require_network,
                          std::string &error);

// Loads and parses the same architecture YAML file passed to Neura.  The
// network section is independent from Neura's per-CGRA link_defaults section.
FailureOr<std::optional<InterTaskNetworkSpec>>
loadInterTaskNetworkSpec(llvm::StringRef architecture_path,
                         bool require_network, std::string &error);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_JOINT_SCHEDULING_INTER_TASK_NETWORK_H
