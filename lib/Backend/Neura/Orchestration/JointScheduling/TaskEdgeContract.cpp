//===- TaskEdgeContract.cpp ----------------------------------*- C++ -*-===//
//
// Implements the typed task-edge and payload contracts used by joint
// scheduling. This file deliberately has no scheduler integration.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"

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
  TaskEdgeOrigin origin = TaskEdgeOrigin::Taskflow;

  bool operator<(const EdgeIdentity &other) const {
    return std::tie(producer_task, consumer_task, kind, producer_segment,
                    producer_index, consumer_segment, consumer_index, origin) <
           std::tie(other.producer_task, other.consumer_task, other.kind,
                    other.producer_segment, other.producer_index,
                    other.consumer_segment, other.consumer_index, other.origin);
  }
};

struct ProducerResult {
  TaskflowTaskOp task;
  TaskResultSegment segment = TaskResultSegment::None;
  uint32_t index = 0;
  Value value;
};

struct StaticRegion {
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

struct ProducerWriteRegion {
  Value base;
  StaticRegion region;
};

struct ConsumerInputRegion {
  bool descriptor_present = false;
  std::optional<StaticRegion> region;
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

TaskflowJoinOp findForwardedJoin(Value value) {
  while (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    value = channel.getSource();
  return value.getDefiningOp<TaskflowJoinOp>();
}

bool validateRegion(const StaticRegion &region, ShapedType type) {
  if (!type.hasStaticShape() ||
      region.lower.size() != static_cast<size_t>(type.getRank()) ||
      region.upper.size() != region.lower.size())
    return false;
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension)
    if (region.lower[dimension] < 0 ||
        region.upper[dimension] <= region.lower[dimension] ||
        region.upper[dimension] > type.getShape()[dimension])
      return false;
  return true;
}

std::optional<ProducerWriteRegion>
getProducerWriteRegion(ProducerResult producer) {
  if (producer.segment != TaskResultSegment::DoneWrites ||
      !producer.task.getBody().hasOneBlock())
    return std::nullopt;
  auto type = dyn_cast<ShapedType>(producer.value.getType());
  auto taskYield = dyn_cast<TaskflowYieldOp>(
      producer.task.getBody().front().getTerminator());
  if (!type || !taskYield || producer.index >= taskYield.getDoneWrites().size())
    return std::nullopt;
  auto argument =
      dyn_cast<BlockArgument>(taskYield.getDoneWrites()[producer.index]);
  unsigned firstWrite = producer.task.getWillReads().size();
  if (!argument || argument.getOwner() != &producer.task.getBody().front() ||
      argument.getArgNumber() < firstWrite)
    return std::nullopt;
  unsigned writeIndex = argument.getArgNumber() - firstWrite;
  size_t outputCount = producer.task.getWillWrites().size();
  auto lowers =
      producer.task->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
  auto uppers =
      producer.task->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
  if (writeIndex >= outputCount || !lowers || !uppers ||
      lowers.size() != outputCount || uppers.size() != outputCount ||
      producer.task.getOriginalWriteMemrefs().size() != outputCount)
    return std::nullopt;
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[writeIndex]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[writeIndex]);
  if (!lower || !upper)
    return std::nullopt;
  StaticRegion region{SmallVector<int64_t>(lower.asArrayRef()),
                      SmallVector<int64_t>(upper.asArrayRef())};
  if (!validateRegion(region, type))
    return std::nullopt;
  return ProducerWriteRegion{
      producer.task.getOriginalWriteMemrefs()[writeIndex], std::move(region)};
}

// The dependence-state SSA value yielded by a producer is not itself the
// original memref identity recorded on the consumer.  In particular, a
// materialized private partial buffer is yielded by the producer and threaded
// directly into the consumer, while the consumer's original_read_memrefs may
// still name that yielded state.  Resolve those state values back to the
// original write/read operand so region proofs compare storage roots rather
// than incidental SSA versions.
static FailureOr<Value> resolveOriginalMemrefRoot(Value value,
                                                  DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return failure();
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    return resolveOriginalMemrefRoot(channel.getSource(), visited);
  if (auto join = value.getDefiningOp<TaskflowJoinOp>())
    return join.getBase();
  auto task = value.getDefiningOp<TaskflowTaskOp>();
  if (!task)
    return value;
  auto yield =
      dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
  if (!yield)
    return failure();
  auto resolveSegment = [&](ValueRange results, ValueRange originals,
                            bool requireWrite) -> FailureOr<Value> {
    auto result = llvm::find(results, value);
    if (result == results.end())
      return failure();
    unsigned index = result - results.begin();
    if (requireWrite) {
      if (index >= yield.getDoneWrites().size() || index >= originals.size())
        return failure();
      auto argument = dyn_cast<BlockArgument>(yield.getDoneWrites()[index]);
      unsigned firstWrite = task.getWillReads().size();
      if (!argument || argument.getOwner() != &task.getBody().front() ||
          argument.getArgNumber() < firstWrite ||
          argument.getArgNumber() >= firstWrite + task.getWillWrites().size())
        return failure();
      unsigned writeIndex = argument.getArgNumber() - firstWrite;
      if (writeIndex >= originals.size())
        return failure();
      return resolveOriginalMemrefRoot(originals[writeIndex], visited);
    }
    if (index >= yield.getDoneReads().size() || index >= originals.size())
      return failure();
    return resolveOriginalMemrefRoot(originals[index], visited);
  };
  if (auto root = resolveSegment(task.getDoneWrites(),
                                 task.getOriginalWriteMemrefs(), true);
      succeeded(root))
    return *root;
  if (auto root = resolveSegment(task.getDoneReads(),
                                 task.getOriginalReadMemrefs(), false);
      succeeded(root))
    return *root;
  return failure();
}

static bool matchingOriginalMemrefRoot(Value producerRoot, Value consumerRoot) {
  DenseSet<Value> producerVisited;
  DenseSet<Value> consumerVisited;
  FailureOr<Value> resolvedProducer =
      resolveOriginalMemrefRoot(producerRoot, producerVisited);
  FailureOr<Value> resolvedConsumer =
      resolveOriginalMemrefRoot(consumerRoot, consumerVisited);
  return succeeded(resolvedProducer) && succeeded(resolvedConsumer) &&
         *resolvedProducer == *resolvedConsumer;
}

FailureOr<ConsumerInputRegion> getConsumerInputRegion(TaskflowTaskOp consumer,
                                                      uint32_t inputIndex,
                                                      std::string &error) {
  auto lowers =
      consumer->getAttrOfType<ArrayAttr>(kTilingInputRegionLowersAttr);
  auto uppers =
      consumer->getAttrOfType<ArrayAttr>(kTilingInputRegionUppersAttr);
  auto reasons =
      consumer->getAttrOfType<ArrayAttr>(kTilingInputRegionReasonsAttr);
  if (!lowers && !uppers && !reasons)
    return ConsumerInputRegion{};
  size_t inputCount = consumer.getWillReads().size();
  if (!lowers || !uppers || !reasons || lowers.size() != inputCount ||
      uppers.size() != inputCount || reasons.size() != inputCount ||
      consumer.getOriginalReadMemrefs().size() != inputCount ||
      inputIndex >= inputCount) {
    error = "tiled input-region descriptors must contain one lower, upper, "
            "and reason entry per will_reads operand";
    return failure();
  }
  if (!isa<StringAttr>(reasons[inputIndex])) {
    error = "tiled input-region reasons must be string attributes";
    return failure();
  }
  bool unknownLower = isa<UnitAttr>(lowers[inputIndex]);
  bool unknownUpper = isa<UnitAttr>(uppers[inputIndex]);
  if (unknownLower || unknownUpper) {
    if (!unknownLower || !unknownUpper) {
      error = "an unknown tiled input region requires unit lower and upper "
              "entries";
      return failure();
    }
    return ConsumerInputRegion{/*descriptor_present=*/true, std::nullopt};
  }
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[inputIndex]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[inputIndex]);
  auto type =
      dyn_cast<ShapedType>(consumer.getWillReads()[inputIndex].getType());
  if (!lower || !upper || !type) {
    error = "known tiled input regions require shaped operands and dense "
            "integer bounds";
    return failure();
  }
  StaticRegion region{SmallVector<int64_t>(lower.asArrayRef()),
                      SmallVector<int64_t>(upper.asArrayRef())};
  if (!validateRegion(region, type)) {
    error = "tiled input region is outside its static operand shape";
    return failure();
  }
  return ConsumerInputRegion{/*descriptor_present=*/true, std::move(region)};
}

std::optional<StaticRegion> intersectRegions(const StaticRegion &lhs,
                                             const StaticRegion &rhs) {
  if (lhs.lower.size() != rhs.lower.size())
    return std::nullopt;
  StaticRegion intersection;
  for (size_t dimension = 0; dimension < lhs.lower.size(); ++dimension) {
    int64_t lower = std::max(lhs.lower[dimension], rhs.lower[dimension]);
    int64_t upper = std::min(lhs.upper[dimension], rhs.upper[dimension]);
    if (upper <= lower)
      return std::nullopt;
    intersection.lower.push_back(lower);
    intersection.upper.push_back(upper);
  }
  return intersection;
}

bool isContainedIn(const StaticRegion &inner, ArrayRef<int64_t> outerLower,
                   ArrayRef<int64_t> outerUpper) {
  if (inner.lower.size() != outerLower.size() ||
      inner.upper.size() != outerUpper.size())
    return false;
  for (size_t dimension = 0; dimension < inner.lower.size(); ++dimension)
    if (inner.lower[dimension] < outerLower[dimension] ||
        inner.upper[dimension] > outerUpper[dimension])
      return false;
  return true;
}

FailureOr<StaticRegion> getOriginalAccessRegion(TaskflowTaskOp task,
                                                Value memory, bool write) {
  ValueRange roots =
      write ? task.getOriginalWriteMemrefs() : task.getOriginalReadMemrefs();
  auto found = llvm::find(roots, memory);
  if (found == roots.end() ||
      std::find(std::next(found), roots.end(), memory) != roots.end())
    return failure();
  unsigned index = found - roots.begin();
  StringRef lowerName =
      write ? kTilingOutputRegionLowersAttr : kTilingInputRegionLowersAttr;
  StringRef upperName =
      write ? kTilingOutputRegionUppersAttr : kTilingInputRegionUppersAttr;
  auto lowers = task->getAttrOfType<ArrayAttr>(lowerName);
  auto uppers = task->getAttrOfType<ArrayAttr>(upperName);
  if (!lowers || !uppers || lowers.size() != roots.size() ||
      uppers.size() != roots.size())
    return failure();
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[index]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[index]);
  auto type = dyn_cast<ShapedType>(memory.getType());
  if (!lower || !upper || !type)
    return failure();
  StaticRegion region{SmallVector<int64_t>(lower.asArrayRef()),
                      SmallVector<int64_t>(upper.asArrayRef())};
  if (!validateRegion(region, type))
    return failure();
  return region;
}

bool haveDisjointAccessRegions(TaskflowTaskOp writer, TaskflowTaskOp other,
                               bool otherReads, bool otherWrites,
                               Value memory) {
  FailureOr<StaticRegion> writerRegion =
      getOriginalAccessRegion(writer, memory, /*write=*/true);
  if (failed(writerRegion))
    return false;
  auto disjoint = [](const StaticRegion &lhs, const StaticRegion &rhs) {
    if (lhs.lower.size() != rhs.lower.size() ||
        lhs.upper.size() != lhs.lower.size() ||
        rhs.upper.size() != lhs.lower.size())
      return false;
    for (size_t dimension = 0; dimension < lhs.lower.size(); ++dimension)
      if (lhs.upper[dimension] <= rhs.lower[dimension] ||
          rhs.upper[dimension] <= lhs.lower[dimension])
        return true;
    return false;
  };
  if (otherReads) {
    FailureOr<StaticRegion> readRegion =
        getOriginalAccessRegion(other, memory, /*write=*/false);
    if (failed(readRegion) || !disjoint(*writerRegion, *readRegion))
      return false;
  }
  if (otherWrites) {
    FailureOr<StaticRegion> writeRegion =
        getOriginalAccessRegion(other, memory, /*write=*/true);
    if (failed(writeRegion) || !disjoint(*writerRegion, *writeRegion))
      return false;
  }
  return true;
}

std::optional<uint32_t> firstMemoryIndex(ValueRange values, Value memory) {
  auto found = llvm::find(values, memory);
  if (found == values.end())
    return std::nullopt;
  return static_cast<uint32_t>(found - values.begin());
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

enum class PayloadSource : uint8_t {
  Unknown,
  StaticType,
  ProvenTiledRegion,
  Explicit,
};

struct PayloadFact {
  std::optional<uint64_t> bits;
  PayloadSource source = PayloadSource::Unknown;
};

struct ExplicitPayloadFact {
  bool present = false;
  std::optional<uint64_t> bits;
};

PayloadFact getProducerPayloadFact(ProducerResult producer) {
  auto shaped = dyn_cast<ShapedType>(producer.value.getType());
  if (!shaped || producer.segment != TaskResultSegment::DoneWrites) {
    std::optional<uint64_t> bits = getTypePayloadBits(producer.value.getType());
    return {bits, bits ? PayloadSource::StaticType : PayloadSource::Unknown};
  }
  if (!shaped.hasStaticShape())
    return {};
  bool hasTiledRegion = producer.task->hasAttr(kTilingOutputRegionLowersAttr) ||
                        producer.task->hasAttr(kTilingOutputRegionUppersAttr);
  if (hasTiledRegion) {
    std::optional<ProducerWriteRegion> writeRegion =
        getProducerWriteRegion(producer);
    if (!writeRegion)
      return {};
    uint64_t elements = 1;
    for (int64_t dimension = 0; dimension < shaped.getRank(); ++dimension) {
      uint64_t extent =
          static_cast<uint64_t>(writeRegion->region.upper[dimension] -
                                writeRegion->region.lower[dimension]);
      if (elements > UINT64_MAX / extent)
        return {};
      elements *= extent;
    }
    std::optional<uint64_t> elementBits =
        getTypePayloadBits(shaped.getElementType());
    if (!elementBits || elements > UINT64_MAX / *elementBits)
      return {};
    return {elements * *elementBits, PayloadSource::ProvenTiledRegion};
  }

  // Without a compiler-proven tiled region, counter bounds say nothing about
  // which output elements are actually transferred.  The only sound static
  // fact is the complete shaped value; a narrower transfer needs an explicit
  // consumer contract.
  std::optional<uint64_t> bits = getTypePayloadBits(producer.value.getType());
  return {bits, bits ? PayloadSource::StaticType : PayloadSource::Unknown};
}

PayloadFact getRegionPayloadFact(Type valueType, const StaticRegion &region) {
  auto shaped = dyn_cast<ShapedType>(valueType);
  if (!shaped || !validateRegion(region, shaped))
    return {};
  uint64_t elements = 1;
  for (size_t dimension = 0; dimension < region.lower.size(); ++dimension) {
    uint64_t extent = static_cast<uint64_t>(region.upper[dimension] -
                                            region.lower[dimension]);
    if (elements > UINT64_MAX / extent)
      return {};
    elements *= extent;
  }
  std::optional<uint64_t> elementBits =
      getTypePayloadBits(shaped.getElementType());
  if (!elementBits || elements > UINT64_MAX / *elementBits)
    return {};
  return {elements * *elementBits, PayloadSource::ProvenTiledRegion};
}

ExplicitPayloadFact getExplicitPayloadBits(TaskflowTaskOp consumer,
                                           TaskOperandSegment segment,
                                           uint32_t index, std::string &error) {
  auto attribute = consumer->getAttr(kEdgePayloadBitsAttr);
  if (!attribute)
    return {};
  auto payload_bits = dyn_cast<DenseI64ArrayAttr>(attribute);
  if (!payload_bits) {
    error = "amoeba.edge_payload_bits must be a DenseI64ArrayAttr";
    return {true, std::nullopt};
  }

  size_t read_count = consumer.getWillReads().size();
  size_t value_count = consumer.getValueInputs().size();
  if (payload_bits.asArrayRef().size() != read_count + value_count) {
    error = "amoeba.edge_payload_bits must contain one entry for every "
            "will_reads and value_inputs operand";
    return {true, std::nullopt};
  }
  if (segment != TaskOperandSegment::WillReads &&
      segment != TaskOperandSegment::ValueInputs) {
    error = "payload facts are only valid for data-carrying operands";
    return {true, std::nullopt};
  }
  size_t payload_index = index;
  if (segment == TaskOperandSegment::ValueInputs)
    payload_index += read_count;
  int64_t bits = payload_bits[payload_index];
  if (bits < -1) {
    error = "amoeba.edge_payload_bits entries must be non-negative or -1";
    return {true, std::nullopt};
  }
  if (bits == -1)
    return {true, std::nullopt};
  return {true, static_cast<uint64_t>(bits)};
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

StringRef stringifyTaskEdgeOrigin(TaskEdgeOrigin origin) {
  switch (origin) {
  case TaskEdgeOrigin::Taskflow:
    return "taskflow";
  case TaskEdgeOrigin::MemoryOrder:
    return "memory_order";
  case TaskEdgeOrigin::Control:
    return "control";
  }
  llvm_unreachable("unknown task edge origin");
}

StringRef stringifyTaskEdgeScope(TaskEdgeScope scope) {
  switch (scope) {
  case TaskEdgeScope::TensorWide:
    return "tensor_wide";
  case TaskEdgeScope::TileLocal:
    return "tile_local";
  }
  llvm_unreachable("unknown task edge scope");
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
  auto addEdge =
      [&](ProducerResult producer, TaskflowTaskOp consumer, TaskEdgeKind kind,
          TaskOperandSegment consumer_segment, uint32_t consumer_index,
          std::optional<uint64_t> payload, TaskEdgeOrigin origin,
          TaskEdgeScope scope,
          const std::optional<StaticRegion> &transferRegion) -> bool {
    unsigned producer_id = task_ids.lookup(producer.task.getOperation());
    unsigned consumer_id = task_ids.lookup(consumer.getOperation());
    if (producer_id == consumer_id) {
      error = "task edge cannot connect a task to itself";
      return false;
    }
    EdgeIdentity identity{producer_id,      consumer_id,    kind,
                          producer.segment, producer.index, consumer_segment,
                          consumer_index,   origin};
    if (!identities.insert(identity).second) {
      error = "duplicate task edge identity";
      return false;
    }
    if (origin == TaskEdgeOrigin::Taskflow &&
        (kind == TaskEdgeKind::Value || kind == TaskEdgeKind::Raw) &&
        !payload && options.require_payload) {
      error = "data-carrying task edge has unknown payload size";
      return false;
    }
    SmallVector<int64_t> transferLower;
    SmallVector<int64_t> transferUpper;
    if (transferRegion) {
      transferLower = transferRegion->lower;
      transferUpper = transferRegion->upper;
    }
    graph.edges_.push_back({producer.task, consumer, kind, producer.segment,
                            producer.index, consumer_segment, consumer_index,
                            origin, scope, std::move(transferLower),
                            std::move(transferUpper), payload});
    return true;
  };

  auto addOperandEdges = [&](TaskflowTaskOp consumer, ValueRange values,
                             TaskOperandSegment operand_segment,
                             TaskEdgeKind expected_kind) -> bool {
    for (auto indexed : llvm::enumerate(values)) {
      ConsumerInputRegion consumerRegion;
      if (operand_segment == TaskOperandSegment::WillReads) {
        FailureOr<ConsumerInputRegion> parsedRegion =
            getConsumerInputRegion(consumer, indexed.index(), error);
        if (failed(parsedRegion))
          return false;
        consumerRegion = std::move(*parsedRegion);
      }
      SmallVector<ProducerResult> producers;
      DenseSet<Value> visited;
      if (failed(collectProducerResults(
              indexed.value(), task_ids, options.require_payload,
              /*crossedChannel=*/false, producers, visited, error)))
        return false;
      if (producers.empty())
        continue;

      SmallVector<std::optional<StaticRegion>> transferRegions(
          producers.size());
      bool tileLocal = false;
      TaskflowJoinOp join = findForwardedJoin(indexed.value());
      if (operand_segment == TaskOperandSegment::WillReads &&
          consumerRegion.region &&
          indexed.index() < consumer.getOriginalReadMemrefs().size() &&
          producers.size() == 1 && !join) {
        // A direct done_writes -> will_reads edge is tile-local only when the
        // producer's explicit output region and the consumer's per-operand
        // input region refer to the same original memref root.  Unknown or
        // mismatched provenance deliberately falls through to tensor-wide.
        std::optional<ProducerWriteRegion> writeRegion =
            getProducerWriteRegion(producers.front());
        Value consumerRoot = consumer.getOriginalReadMemrefs()[indexed.index()];
        if (writeRegion &&
            matchingOriginalMemrefRoot(writeRegion->base, consumerRoot)) {
          std::optional<StaticRegion> intersection =
              intersectRegions(writeRegion->region, *consumerRegion.region);
          if (!intersection) {
            error = "proven tiled consumer input region does not overlap its "
                    "producer output region";
            return false;
          }
          transferRegions[0] = std::move(intersection);
          tileLocal = true;
        }
      } else if (operand_segment == TaskOperandSegment::WillReads &&
                 consumerRegion.region && join &&
                 indexed.index() < consumer.getOriginalReadMemrefs().size() &&
                 isContainedIn(*consumerRegion.region, join.getRegionLower(),
                               join.getRegionUpper())) {
        Value consumerBase = consumer.getOriginalReadMemrefs()[indexed.index()];
        SmallVector<std::optional<StaticRegion>> candidateRegions;
        candidateRegions.reserve(producers.size());
        bool completeRegionProof = true;
        for (ProducerResult producer : producers) {
          std::optional<ProducerWriteRegion> writeRegion =
              getProducerWriteRegion(producer);
          // The join base is the producer's incoming task state.  For an
          // in-place task this can be another task result rather than the
          // allocation itself.  Original memref provenance on both endpoint
          // tasks is the canonical storage identity, so compare against the
          // consumer root after requiring every joined producer to agree.
          if (!writeRegion ||
              !matchingOriginalMemrefRoot(writeRegion->base, consumerBase)) {
            completeRegionProof = false;
            break;
          }
          candidateRegions.push_back(
              intersectRegions(writeRegion->region, *consumerRegion.region));
        }
        if (completeRegionProof) {
          SmallVector<ProducerResult> intersectingProducers;
          SmallVector<std::optional<StaticRegion>> intersections;
          for (auto [producer, intersection] :
               llvm::zip(producers, candidateRegions)) {
            if (!intersection)
              continue;
            intersectingProducers.push_back(producer);
            intersections.push_back(std::move(intersection));
          }
          if (intersectingProducers.empty()) {
            error = "proven tiled consumer input region has no intersecting "
                    "producer tile";
            return false;
          }
          producers = std::move(intersectingProducers);
          transferRegions = std::move(intersections);
          tileLocal = true;
        }
      }

      struct ResolvedEdge {
        ProducerResult producer;
        TaskEdgeKind kind;
        PayloadFact payload;
        std::optional<StaticRegion> transfer_region;
      };
      SmallVector<ResolvedEdge> resolvedEdges;
      uint64_t inferredPayloadSum = 0;
      bool completePayload = true;
      bool hasDataEdge = false;
      for (auto [producerIndex, producer] : llvm::enumerate(producers)) {
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
          // Older Taskflow producers use a done_reads/done_writes result as
          // an SSA sequencing token in value_inputs.  It carries no scalar
          // value and must not become a communication transport, but it still
          // supplies real precedence.  Preserve that contract as a canonical
          // Taskflow control edge instead of rejecting it or pretending the
          // referenced memory object is value payload.
          kind = TaskEdgeKind::Control;
        }
        PayloadFact payload;
        if (kind == TaskEdgeKind::Value || kind == TaskEdgeKind::Raw) {
          hasDataEdge = true;
          payload = tileLocal
                        ? getRegionPayloadFact(producer.value.getType(),
                                               *transferRegions[producerIndex])
                        : getProducerPayloadFact(producer);
          if (!payload.bits)
            completePayload = false;
          else if (*payload.bits > UINT64_MAX - inferredPayloadSum) {
            error = "joined task payload sum overflows unsigned 64-bit bits";
            return false;
          } else {
            inferredPayloadSum += *payload.bits;
          }
        }
        resolvedEdges.push_back(
            {producer, kind, payload, transferRegions[producerIndex]});
      }

      ExplicitPayloadFact explicitPayload;
      if (hasDataEdge)
        explicitPayload = getExplicitPayloadBits(consumer, operand_segment,
                                                 indexed.index(), error);
      if (!error.empty())
        return false;
      if (explicitPayload.present) {
        if (resolvedEdges.size() == 1) {
          PayloadFact &payload = resolvedEdges.front().payload;
          if (explicitPayload.bits &&
              payload.source == PayloadSource::ProvenTiledRegion &&
              payload.bits && *payload.bits != *explicitPayload.bits) {
            error = "amoeba.edge_payload_bits disagrees with the proven "
                    "transferred region";
            return false;
          }
          payload = {explicitPayload.bits, PayloadSource::Explicit};
        } else if (!explicitPayload.bits) {
          for (ResolvedEdge &resolved : resolvedEdges)
            resolved.payload = {std::nullopt, PayloadSource::Explicit};
        } else if (!completePayload ||
                   inferredPayloadSum != *explicitPayload.bits) {
          error = "amoeba.edge_payload_bits cannot be partitioned across "
                  "taskflow.join producers";
          return false;
        }
      }
      for (const ResolvedEdge &resolved : resolvedEdges)
        if (!addEdge(resolved.producer, consumer, resolved.kind,
                     operand_segment, indexed.index(), resolved.payload.bits,
                     TaskEdgeOrigin::Taskflow,
                     tileLocal ? TaskEdgeScope::TileLocal
                               : TaskEdgeScope::TensorWide,
                     resolved.transfer_region))
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
                   TaskOperandSegment::Control, indexed.index(), std::nullopt,
                   TaskEdgeOrigin::Control, TaskEdgeScope::TensorWide,
                   std::nullopt))
        return failure();
    }
  }

  // Adds source-order memory precedence to the same canonical edge graph used
  // by scheduling, priority, diagnostics, and trace serialization. An existing
  // edge in the required direction already supplies precedence and is not
  // duplicated. Ordering-only edges never become communication transports.
  std::set<std::pair<unsigned, unsigned>> precedencePairs;
  for (const TaskEdge &edge : graph.edges_) {
    TaskflowTaskOp producer = edge.producer;
    TaskflowTaskOp consumer = edge.consumer;
    precedencePairs.insert({task_ids.lookup(producer.getOperation()),
                            task_ids.lookup(consumer.getOperation())});
  }

  SmallVector<Value> memories;
  for (TaskflowTaskOp task : graph.tasks_) {
    for (Value memory : task.getOriginalReadMemrefs())
      if (!llvm::is_contained(memories, memory))
        memories.push_back(memory);
    for (Value memory : task.getOriginalWriteMemrefs())
      if (!llvm::is_contained(memories, memory))
        memories.push_back(memory);
  }
  for (Value memory : memories) {
    std::optional<std::pair<TaskflowTaskOp, uint32_t>> lastWriter;
    SmallVector<std::pair<TaskflowTaskOp, uint32_t>> readersSinceWrite;
    for (TaskflowTaskOp task : graph.tasks_) {
      std::optional<uint32_t> readIndex =
          firstMemoryIndex(task.getOriginalReadMemrefs(), memory);
      std::optional<uint32_t> writeIndex =
          firstMemoryIndex(task.getOriginalWriteMemrefs(), memory);
      const bool reads = readIndex.has_value();
      const bool writes = writeIndex.has_value();
      if (!reads && !writes)
        continue;

      auto addMemoryOrderEdge = [&](TaskflowTaskOp producer,
                                    uint32_t producerIndex, TaskEdgeKind kind,
                                    TaskOperandSegment consumerSegment,
                                    uint32_t consumerIndex) -> bool {
        unsigned producerId = task_ids.lookup(producer.getOperation());
        unsigned consumerId = task_ids.lookup(task.getOperation());
        if (precedencePairs.count({producerId, consumerId}))
          return true;
        ProducerResult source{producer, TaskResultSegment::None, producerIndex,
                              memory};
        if (!addEdge(source, task, kind, consumerSegment, consumerIndex,
                     std::nullopt, TaskEdgeOrigin::MemoryOrder,
                     TaskEdgeScope::TensorWide, std::nullopt))
          return false;
        precedencePairs.insert({producerId, consumerId});
        return true;
      };

      if (lastWriter && !haveDisjointAccessRegions(lastWriter->first, task,
                                                   reads, writes, memory)) {
        if (!addMemoryOrderEdge(lastWriter->first, lastWriter->second,
                                reads ? TaskEdgeKind::Raw : TaskEdgeKind::Waw,
                                reads ? TaskOperandSegment::WillReads
                                      : TaskOperandSegment::WillWrites,
                                reads ? *readIndex : *writeIndex))
          return failure();
      }
      if (writes) {
        for (auto [reader, readerIndex] : readersSinceWrite) {
          if (haveDisjointAccessRegions(task, reader, /*otherReads=*/true,
                                        /*otherWrites=*/false, memory))
            continue;
          if (!addMemoryOrderEdge(reader, readerIndex, TaskEdgeKind::War,
                                  TaskOperandSegment::WillWrites, *writeIndex))
            return failure();
        }
        readersSinceWrite.clear();
        lastWriter = std::make_pair(task, *writeIndex);
      } else {
        readersSinceWrite.push_back({task, *readIndex});
      }
    }
  }
  return graph;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
