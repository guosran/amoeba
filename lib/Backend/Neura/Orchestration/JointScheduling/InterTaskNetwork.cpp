//===- InterTaskNetwork.cpp ----------------------------------*- C++ -*-===//
//
// Implements the versioned architecture-level inter-task network contract.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

using namespace mlir;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {
namespace {

namespace yaml = llvm::yaml;

constexpr llvm::StringLiteral kInterTaskNetwork = "inter_task_network";
constexpr llvm::StringLiteral kVersion = "version";
constexpr llvm::StringLiteral kRows = "rows";
constexpr llvm::StringLiteral kColumns = "columns";
constexpr llvm::StringLiteral kLocalBandwidth =
    "local_bandwidth_bits_per_cycle";
constexpr llvm::StringLiteral kLinks = "links";
constexpr llvm::StringLiteral kSourceRow = "src_row";
constexpr llvm::StringLiteral kSourceColumn = "src_column";
constexpr llvm::StringLiteral kDestinationRow = "dst_row";
constexpr llvm::StringLiteral kDestinationColumn = "dst_column";
constexpr llvm::StringLiteral kLatency = "latency_cycles";
constexpr llvm::StringLiteral kBandwidth = "bandwidth_bits_per_cycle";

using yaml::MappingNode;
using yaml::ScalarNode;
using yaml::SequenceNode;

bool parseUnsignedScalar(const yaml::Node *node, uint64_t &value) {
  auto *scalar = dyn_cast_or_null<ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> storage;
  llvm::StringRef text = scalar->getValue(storage);
  return !text.empty() && !text.getAsInteger(10, value);
}

bool parseSignedScalar(const yaml::Node *node, int64_t &value) {
  auto *scalar = dyn_cast_or_null<ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> storage;
  llvm::StringRef text = scalar->getValue(storage);
  return !text.empty() && !text.getAsInteger(10, value);
}

bool parseKey(const yaml::Node *node, llvm::StringRef expected) {
  auto *scalar = dyn_cast_or_null<ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> storage;
  return scalar->getValue(storage) == expected;
}

bool isInBounds(TaskNetworkCoordinate coordinate, int rows, int columns) {
  return coordinate.row >= 0 && coordinate.row < rows &&
         coordinate.column >= 0 && coordinate.column < columns;
}

struct CoordinateLess {
  bool operator()(TaskNetworkCoordinate lhs, TaskNetworkCoordinate rhs) const {
    if (lhs.row != rhs.row)
      return lhs.row < rhs.row;
    return lhs.column < rhs.column;
  }
};

struct CoordinatePairLess {
  bool
  operator()(const std::pair<TaskNetworkCoordinate, TaskNetworkCoordinate> &lhs,
             const std::pair<TaskNetworkCoordinate, TaskNetworkCoordinate> &rhs)
      const {
    CoordinateLess less;
    if (less(lhs.first, rhs.first))
      return true;
    if (less(rhs.first, lhs.first))
      return false;
    return less(lhs.second, rhs.second);
  }
};

FailureOr<llvm::SmallVector<InterTaskNetworkLink>>
parseNetworkLinks(SequenceNode *links_sequence, std::string &error) {
  llvm::SmallVector<InterTaskNetworkLink> links;
  for (yaml::Node &node : *links_sequence) {
    auto *link_map = dyn_cast<MappingNode>(&node);
    if (!link_map) {
      error = "inter_task_network link must be a mapping";
      return failure();
    }

    int64_t source_row = 0;
    int64_t source_column = 0;
    int64_t destination_row = 0;
    int64_t destination_column = 0;
    uint64_t latency = 0;
    uint64_t bandwidth = 0;
    bool has_source_row = false;
    bool has_source_column = false;
    bool has_destination_row = false;
    bool has_destination_column = false;
    bool has_latency = false;
    bool has_bandwidth = false;

    for (auto &entry : *link_map) {
      auto *key = dyn_cast_or_null<ScalarNode>(entry.getKey());
      if (!key) {
        error = "inter_task_network link contains a non-scalar key";
        return failure();
      }
      llvm::SmallString<64> key_storage;
      llvm::StringRef key_text = key->getValue(key_storage);
      if (key_text == kSourceRow) {
        if (has_source_row ||
            !parseSignedScalar(entry.getValue(), source_row)) {
          error = "inter_task_network src_row must be an integer";
          return failure();
        }
        has_source_row = true;
      } else if (key_text == kSourceColumn) {
        if (has_source_column ||
            !parseSignedScalar(entry.getValue(), source_column)) {
          error = "inter_task_network src_column must be an integer";
          return failure();
        }
        has_source_column = true;
      } else if (key_text == kDestinationRow) {
        if (has_destination_row ||
            !parseSignedScalar(entry.getValue(), destination_row)) {
          error = "inter_task_network dst_row must be an integer";
          return failure();
        }
        has_destination_row = true;
      } else if (key_text == kDestinationColumn) {
        if (has_destination_column ||
            !parseSignedScalar(entry.getValue(), destination_column)) {
          error = "inter_task_network dst_column must be an integer";
          return failure();
        }
        has_destination_column = true;
      } else if (key_text == kLatency) {
        if (has_latency || !parseUnsignedScalar(entry.getValue(), latency)) {
          error = "inter_task_network latency_cycles must be a "
                  "non-negative integer";
          return failure();
        }
        has_latency = true;
      } else if (key_text == kBandwidth) {
        if (has_bandwidth ||
            !parseUnsignedScalar(entry.getValue(), bandwidth)) {
          error = "inter_task_network bandwidth_bits_per_cycle must be a "
                  "non-negative integer";
          return failure();
        }
        has_bandwidth = true;
      } else {
        error = "unknown inter_task_network link key: " + key_text.str();
        return failure();
      }
    }
    if (!has_source_row || !has_source_column || !has_destination_row ||
        !has_destination_column || !has_latency || !has_bandwidth) {
      error = "inter_task_network link requires source, destination, latency, "
              "and bandwidth";
      return failure();
    }
    if (source_row < 0 || source_column < 0 || destination_row < 0 ||
        destination_column < 0 ||
        source_row > std::numeric_limits<int>::max() ||
        source_column > std::numeric_limits<int>::max() ||
        destination_row > std::numeric_limits<int>::max() ||
        destination_column > std::numeric_limits<int>::max()) {
      error = "inter_task_network link coordinate is outside the supported "
              "range";
      return failure();
    }
    links.push_back(
        {{static_cast<int>(source_row), static_cast<int>(source_column)},
         {static_cast<int>(destination_row),
          static_cast<int>(destination_column)},
         latency,
         bandwidth});
  }
  return links;
}

FailureOr<InterTaskNetworkSpec> parseNetworkMapping(MappingNode *network_map,
                                                    std::string &error) {
  uint64_t version = 0;
  int64_t rows = 0;
  int64_t columns = 0;
  uint64_t local_bandwidth = 0;
  bool has_version = false;
  bool has_rows = false;
  bool has_columns = false;
  bool has_local_bandwidth = false;
  bool has_links = false;
  llvm::SmallVector<InterTaskNetworkLink> links;

  for (auto &entry : *network_map) {
    auto *key = dyn_cast_or_null<ScalarNode>(entry.getKey());
    if (!key) {
      error = "inter_task_network contains a non-scalar key";
      return failure();
    }
    llvm::SmallString<64> key_storage;
    llvm::StringRef key_text = key->getValue(key_storage);
    if (key_text == kVersion) {
      if (has_version || !parseUnsignedScalar(entry.getValue(), version)) {
        error = "inter_task_network.version must be a non-negative integer";
        return failure();
      }
      has_version = true;
    } else if (key_text == kRows) {
      if (has_rows || !parseSignedScalar(entry.getValue(), rows)) {
        error = "inter_task_network.rows must be an integer";
        return failure();
      }
      has_rows = true;
    } else if (key_text == kColumns) {
      if (has_columns || !parseSignedScalar(entry.getValue(), columns)) {
        error = "inter_task_network.columns must be an integer";
        return failure();
      }
      has_columns = true;
    } else if (key_text == kLocalBandwidth) {
      if (has_local_bandwidth ||
          !parseUnsignedScalar(entry.getValue(), local_bandwidth)) {
        error = "inter_task_network.local_bandwidth_bits_per_cycle must be "
                "a non-negative integer";
        return failure();
      }
      has_local_bandwidth = true;
    } else if (key_text == kLinks) {
      if (has_links) {
        error = "inter_task_network.links may only appear once";
        return failure();
      }
      auto *links_sequence = dyn_cast<SequenceNode>(entry.getValue());
      if (!links_sequence) {
        error = "inter_task_network.links must be a sequence";
        return failure();
      }
      FailureOr<llvm::SmallVector<InterTaskNetworkLink>> parsed_links =
          parseNetworkLinks(links_sequence, error);
      if (failed(parsed_links))
        return failure();
      links = std::move(*parsed_links);
      has_links = true;
    } else {
      error = "unknown inter_task_network key: " + key_text.str();
      return failure();
    }
  }

  if (!has_version || !has_rows || !has_columns || !has_local_bandwidth ||
      !has_links) {
    error = "inter_task_network requires version, rows, columns, "
            "local_bandwidth_bits_per_cycle, and links";
    return failure();
  }
  if (rows <= 0 || columns <= 0) {
    error = "inter_task_network dimensions must be positive";
    return failure();
  }
  if (rows > std::numeric_limits<int>::max() ||
      columns > std::numeric_limits<int>::max()) {
    error = "inter_task_network dimensions exceed the supported range";
    return failure();
  }

  InterTaskNetworkSpec network(version, static_cast<int>(rows),
                               static_cast<int>(columns), local_bandwidth,
                               std::move(links));
  if (!network.validateForGrid(network.getRows(), network.getColumns(), error))
    return failure();
  return network;
}

FailureOr<std::optional<InterTaskNetworkSpec>>
parseNetworkDocument(llvm::StringRef yaml_text, bool require_network,
                     std::string &error) {
  llvm::SourceMgr source_manager;
  llvm::yaml::Stream stream(yaml_text, source_manager);
  auto document = stream.begin();
  if (document == stream.end() || stream.failed()) {
    error = "cannot parse architecture YAML";
    return failure();
  }
  auto *root = document->getRoot();
  auto *root_map = dyn_cast_or_null<MappingNode>(root);
  if (!root_map) {
    error = "architecture YAML root is not a mapping";
    return failure();
  }

  std::optional<InterTaskNetworkSpec> network;
  for (auto &entry : *root_map) {
    if (!parseKey(entry.getKey(), kInterTaskNetwork))
      continue;
    if (network) {
      error = "architecture YAML contains duplicate inter_task_network "
              "sections";
      return failure();
    }
    auto *network_map = dyn_cast<MappingNode>(entry.getValue());
    if (!network_map) {
      error = "inter_task_network must be a mapping";
      return failure();
    }
    FailureOr<InterTaskNetworkSpec> parsed =
        parseNetworkMapping(network_map, error);
    if (failed(parsed))
      return failure();
    network = std::move(*parsed);
  }
  if (!network) {
    if (require_network) {
      error = "explicit communication mode requires an inter_task_network "
              "section";
      return failure();
    }
    return std::optional<InterTaskNetworkSpec>();
  }
  return network;
}

} // namespace

bool InterTaskNetworkSpec::validateForGrid(int grid_rows, int grid_columns,
                                           std::string &error) const {
  error.clear();
  if (version_ != kSchemaVersion) {
    error = "unsupported inter_task_network schema version";
    return false;
  }
  if (rows_ <= 0 || columns_ <= 0 || grid_rows <= 0 || grid_columns <= 0) {
    error = "inter_task_network dimensions must be positive";
    return false;
  }
  if (rows_ != grid_rows || columns_ != grid_columns) {
    error = "inter_task_network dimensions do not match the multi-CGRA grid";
    return false;
  }
  if (local_bandwidth_bits_per_cycle_ == 0) {
    error = "inter_task_network local bandwidth must be positive";
    return false;
  }

  std::set<std::pair<TaskNetworkCoordinate, TaskNetworkCoordinate>,
           CoordinatePairLess>
      seen_links;
  for (const InterTaskNetworkLink &link : links_) {
    if (!isInBounds(link.source, rows_, columns_) ||
        !isInBounds(link.destination, rows_, columns_)) {
      error = "inter_task_network link endpoint is outside the grid";
      return false;
    }
    if (link.source == link.destination) {
      error = "inter_task_network link cannot be self-looping";
      return false;
    }
    if (link.latency_cycles == 0) {
      error = "inter_task_network link latency must be positive";
      return false;
    }
    if (link.bandwidth_bits_per_cycle == 0) {
      error = "inter_task_network link bandwidth must be positive";
      return false;
    }
    if (!seen_links.insert({link.source, link.destination}).second) {
      error = "inter_task_network contains duplicate directed links";
      return false;
    }
  }
  return true;
}

FailureOr<InterTaskNetworkPath>
InterTaskNetworkSpec::route(TaskNetworkCoordinate source,
                            TaskNetworkCoordinate destination,
                            std::string &error) const {
  error.clear();
  if (!validateForGrid(rows_, columns_, error))
    return failure();
  if (!isInBounds(source, rows_, columns_) ||
      !isInBounds(destination, rows_, columns_)) {
    error = "inter_task_network route endpoint is outside the grid";
    return failure();
  }

  InterTaskNetworkPath path;
  path.source = source;
  path.destination = destination;
  if (source == destination) {
    path.bottleneck_bandwidth_bits_per_cycle = local_bandwidth_bits_per_cycle_;
    return path;
  }

  std::map<TaskNetworkCoordinate, TaskNetworkCoordinate, CoordinateLess>
      predecessor;
  std::set<TaskNetworkCoordinate, CoordinateLess> visited;
  llvm::SmallVector<TaskNetworkCoordinate> queue;
  queue.push_back(source);
  visited.insert(source);
  size_t next = 0;
  while (next < queue.size()) {
    TaskNetworkCoordinate current = queue[next++];
    llvm::SmallVector<const InterTaskNetworkLink *> neighbors;
    for (const InterTaskNetworkLink &link : links_)
      if (link.source == current && !visited.count(link.destination))
        neighbors.push_back(&link);
    llvm::sort(neighbors, [](const InterTaskNetworkLink *lhs,
                             const InterTaskNetworkLink *rhs) {
      return CoordinateLess{}(lhs->destination, rhs->destination);
    });
    for (const InterTaskNetworkLink *link : neighbors) {
      visited.insert(link->destination);
      predecessor[link->destination] = current;
      queue.push_back(link->destination);
      if (link->destination == destination)
        break;
    }
    if (visited.count(destination))
      break;
  }
  if (!visited.count(destination)) {
    error = "inter_task_network has no directed path between task endpoints";
    return failure();
  }

  llvm::SmallVector<const InterTaskNetworkLink *> reverse_links;
  TaskNetworkCoordinate current = destination;
  while (!(current == source)) {
    auto predecessor_it = predecessor.find(current);
    if (predecessor_it == predecessor.end()) {
      error = "inter_task_network route reconstruction failed";
      return failure();
    }
    TaskNetworkCoordinate previous = predecessor_it->second;
    const InterTaskNetworkLink *found = nullptr;
    for (const InterTaskNetworkLink &link : links_) {
      if (link.source == previous && link.destination == current) {
        found = &link;
        break;
      }
    }
    if (!found) {
      error = "inter_task_network route reconstruction failed";
      return failure();
    }
    reverse_links.push_back(found);
    current = previous;
  }
  for (auto it = reverse_links.rbegin(); it != reverse_links.rend(); ++it)
    path.links.push_back(**it);

  path.bottleneck_bandwidth_bits_per_cycle = UINT64_MAX;
  for (const InterTaskNetworkLink &link : path.links) {
    if (path.latency_cycles > UINT64_MAX - link.latency_cycles) {
      error = "inter_task_network path latency overflows unsigned 64-bit "
              "cycles";
      return failure();
    }
    path.latency_cycles += link.latency_cycles;
    path.bottleneck_bandwidth_bits_per_cycle =
        std::min(path.bottleneck_bandwidth_bits_per_cycle,
                 link.bandwidth_bits_per_cycle);
  }
  return path;
}

FailureOr<int64_t> InterTaskNetworkSpec::transferCycles(
    TaskNetworkCoordinate source, TaskNetworkCoordinate destination,
    uint64_t payload_bits, std::string &error) const {
  error.clear();
  FailureOr<InterTaskNetworkPath> path = route(source, destination, error);
  if (failed(path))
    return failure();
  uint64_t bandwidth = path->bottleneck_bandwidth_bits_per_cycle;
  if (bandwidth == 0) {
    error = "inter_task_network path has no positive bandwidth";
    return failure();
  }
  uint64_t transmission_cycles =
      payload_bits == 0 ? 0 : ((payload_bits - 1) / bandwidth) + 1;
  if (path->latency_cycles > UINT64_MAX - transmission_cycles) {
    error = "inter_task_network transfer cycles overflow unsigned 64-bit "
            "cycles";
    return failure();
  }
  uint64_t cycles = path->latency_cycles + transmission_cycles;
  if (cycles > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    error = "inter_task_network transfer cycles exceed signed 64-bit cycles";
    return failure();
  }
  return static_cast<int64_t>(cycles);
}

FailureOr<std::optional<InterTaskNetworkSpec>>
parseInterTaskNetworkYaml(llvm::StringRef yaml, bool require_network,
                          std::string &error) {
  error.clear();
  return parseNetworkDocument(yaml, require_network, error);
}

FailureOr<std::optional<InterTaskNetworkSpec>>
loadInterTaskNetworkSpec(llvm::StringRef architecture_path,
                         bool require_network, std::string &error) {
  error.clear();
  if (architecture_path.empty()) {
    if (require_network) {
      error = "explicit communication mode requires an architecture path";
      return failure();
    }
    return std::optional<InterTaskNetworkSpec>();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(architecture_path);
  if (!buffer) {
    error = "cannot read architecture specification " +
            architecture_path.str() + ": " + buffer.getError().message();
    return failure();
  }
  return parseNetworkDocument((*buffer)->getBuffer(), require_network, error);
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
