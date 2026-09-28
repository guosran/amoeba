//===- ExtractJointTaskGraphFactsPass.cpp ---------------------*- C++ -*-===//
//
// Emits deterministic Taskflow graph facts and conservative legal rewrite
// actions. The graph-variant driver invokes this pass again after every
// rewrite; no legality decision is carried across a changed graph.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kFactsSchema = "amoeba-joint-task-graph-facts-v1";

struct EdgeFact {
  unsigned source = 0;
  unsigned target = 0;
  std::string kind;
  std::string producerSegment;
  uint32_t producerIndex = 0;
  std::string consumerSegment;
  uint32_t consumerIndex = 0;
  std::string origin;
  std::string scope;
  std::string semanticRole;
  std::string semanticScope;
  SmallVector<int64_t> regionLower;
  SmallVector<int64_t> regionUpper;
  std::optional<uint64_t> payloadBits;

  bool operator<(const EdgeFact &other) const {
    return std::tie(source, target, kind, producerSegment, producerIndex,
                    consumerSegment, consumerIndex, origin, scope, semanticRole,
                    semanticScope, regionLower, regionUpper, payloadBits) <
           std::tie(other.source, other.target, other.kind,
                    other.producerSegment, other.producerIndex,
                    other.consumerSegment, other.consumerIndex, other.origin,
                    other.scope, other.semanticRole, other.semanticScope,
                    other.regionLower, other.regionUpper, other.payloadBits);
  }
  bool operator==(const EdgeFact &other) const {
    return source == other.source && target == other.target &&
           kind == other.kind && producerSegment == other.producerSegment &&
           producerIndex == other.producerIndex &&
           consumerSegment == other.consumerSegment &&
           consumerIndex == other.consumerIndex && origin == other.origin &&
           scope == other.scope && semanticRole == other.semanticRole &&
           semanticScope == other.semanticScope &&
           regionLower == other.regionLower &&
           regionUpper == other.regionUpper && payloadBits == other.payloadBits;
  }
};

constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

static FailureOr<SmallVector<std::pair<int64_t, int64_t>>>
loopDomain(TaskflowTaskOp task) {
  SmallVector<std::pair<int64_t, int64_t>> domain;
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (!counters.empty()) {
    for (TaskflowCounterOp counter : counters) {
      std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
      std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
      std::optional<int64_t> step = constantIndex(counter.getStep());
      if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
        return failure();
      domain.push_back({*lower, *upper});
    }
    return domain;
  }
  task.walk([&](scf::ForOp loop) {
    std::optional<int64_t> lower = constantIndex(loop.getLowerBound());
    std::optional<int64_t> upper = constantIndex(loop.getUpperBound());
    std::optional<int64_t> step = constantIndex(loop.getStep());
    if (lower && upper && step && *step == 1 && *upper > *lower)
      domain.push_back({*lower, *upper});
  });
  if (domain.empty()) {
    task.walk([&](affine::AffineForOp loop) {
      if (loop.hasConstantBounds() && loop.getStepAsInt() == 1 &&
          loop.getConstantUpperBound() > loop.getConstantLowerBound())
        domain.push_back(
            {loop.getConstantLowerBound(), loop.getConstantUpperBound()});
    });
  }
  if (domain.empty())
    return failure();
  return domain;
}

static Value peelChannel(Value value) {
  while (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    value = channel.getSource();
  return value;
}

static void
sourceTaskIndices(Value value, const DenseMap<Operation *, unsigned> &indices,
                  SmallVectorImpl<std::pair<unsigned, std::string>> &sources,
                  DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    sourceTaskIndices(channel.getSource(), indices, sources, visited);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates())
      sourceTaskIndices(state, indices, sources, visited);
    return;
  }
  auto task = value.getDefiningOp<TaskflowTaskOp>();
  if (!task)
    return;
  auto found = indices.find(task.getOperation());
  if (found == indices.end())
    return;
  std::string kind;
  if (llvm::is_contained(task.getDoneWrites(), value))
    kind = "raw";
  else if (llvm::is_contained(task.getDoneReads(), value))
    kind = "war";
  else if (llvm::is_contained(task.getValueOutputs(), value))
    kind = "value";
  else
    return;
  sources.push_back({found->second, std::move(kind)});
}

static FailureOr<SmallVector<EdgeFact, 0>>
collectEdges(func::FuncOp function,
             const DenseMap<Operation *, unsigned> &indices,
             std::string &error) {
  SmallVector<EdgeFact, 0> edges;
  TaskEdgeGraphOptions options;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(function, options, error);
  if (failed(graph))
    return failure();
  for (const TaskEdge &edge : graph->getEdges()) {
    TaskflowTaskOp producerTask = edge.producer;
    TaskflowTaskOp consumerTask = edge.consumer;
    auto producer = indices.find(producerTask.getOperation());
    auto consumer = indices.find(consumerTask.getOperation());
    if (producer == indices.end() || consumer == indices.end()) {
      error = "typed task edge references a task outside graph metadata";
      return failure();
    }
    edges.push_back({producer->second, consumer->second,
                     stringifyTaskEdgeKind(edge.kind).str(),
                     stringifyTaskResultSegment(edge.producer_segment).str(),
                     edge.producer_index,
                     stringifyTaskOperandSegment(edge.consumer_segment).str(),
                     edge.consumer_index,
                     stringifyTaskEdgeOrigin(edge.origin).str(),
                     stringifyTaskEdgeScope(edge.scope).str(), "", "",
                     edge.transfer_region_lower, edge.transfer_region_upper,
                     edge.payload_bits});
  }
  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

static bool parseSemanticIncomingEdge(StringRef encoded, StringRef &source,
                                      StringRef &role, StringRef &scope) {
  size_t first = encoded.find('|');
  if (first == StringRef::npos)
    return false;
  size_t second = encoded.find('|', first + 1);
  if (second == StringRef::npos ||
      encoded.find('|', second + 1) != StringRef::npos)
    return false;
  source = encoded.take_front(first);
  role = encoded.slice(first + 1, second);
  scope = encoded.drop_front(second + 1);
  return !source.empty() && !role.empty() && !scope.empty();
}

// Semantic annotations describe the producer-facing meaning of an incoming
// edge.  They are deliberately checked against the already-derived Taskflow
// graph so they cannot manufacture dependencies or silently relabel a missing
// one.  The generic edge kind remains raw/war/waw/value/control; semantic_role
// and semantic_scope are additional facts for clients that understand the
// producer's algorithm.
static bool validateSemanticIncomingEdges(
    func::FuncOp function, const DenseMap<Operation *, unsigned> &indices,
    MutableArrayRef<EdgeFact> edges, std::string &error) {
  llvm::StringMap<SmallVector<unsigned>> taskNameIds;
  DenseMap<unsigned, TaskflowTaskOp> tasksById;
  function.walk([&](TaskflowTaskOp task) {
    auto found = indices.find(task.getOperation());
    if (found != indices.end()) {
      taskNameIds[task.getTaskName()].push_back(found->second);
      tasksById[found->second] = task;
    }
  });

  for (TaskflowTaskOp consumer : function.getOps<TaskflowTaskOp>()) {
    auto targetIt = indices.find(consumer.getOperation());
    if (targetIt == indices.end()) {
      error = "semantic edge annotation references a task outside graph "
              "metadata";
      return false;
    }
    unsigned target = targetIt->second;
    Attribute attribute = consumer->getAttr(kSemanticIncomingEdgesAttr);
    if (!attribute)
      continue;
    auto annotations = dyn_cast<ArrayAttr>(attribute);
    if (!annotations) {
      error = "amoeba.semantic.incoming_edges must be an array attribute";
      return false;
    }

    DenseSet<unsigned> annotatedEdges;
    for (Attribute entry : annotations) {
      auto string = dyn_cast<StringAttr>(entry);
      if (!string) {
        error = "amoeba.semantic.incoming_edges entries must be strings";
        return false;
      }
      StringRef sourceName;
      StringRef role;
      StringRef scope;
      if (!parseSemanticIncomingEdge(string.getValue(), sourceName, role,
                                     scope)) {
        error = "amoeba.semantic.incoming_edges entries must have the form "
                "source|role|scope with three non-empty fields";
        return false;
      }
      auto sourceIds = taskNameIds.lookup(sourceName);
      if (sourceIds.size() != 1) {
        error = sourceIds.empty()
                    ? "semantic incoming edge source task does not exist: " +
                          sourceName.str()
                    : "semantic incoming edge source task is ambiguous: " +
                          sourceName.str();
        return false;
      }
      unsigned source = sourceIds.front();
      SmallVector<unsigned> matches;
      for (auto [edgeIndex, edge] : llvm::enumerate(edges))
        if (edge.target == target && edge.source == source &&
            edge.origin == "taskflow")
          matches.push_back(edgeIndex);
      if (matches.empty()) {
        error = "semantic incoming edge does not match an actual Taskflow "
                "edge: " +
                string.getValue().str();
        return false;
      }
      if (matches.size() != 1) {
        error = "semantic incoming edge source is ambiguous for actual "
                "Taskflow edges: " +
                string.getValue().str();
        return false;
      }
      unsigned edgeIndex = matches.front();
      if (!annotatedEdges.insert(edgeIndex).second) {
        error = "duplicate semantic incoming edge annotation: " +
                string.getValue().str();
        return false;
      }
      EdgeFact &edge = edges[edgeIndex];
      if (scope != edge.scope) {
        error = "semantic incoming edge scope disagrees with the actual "
                "Taskflow edge: " +
                string.getValue().str();
        return false;
      }
      edge.semanticRole = role.str();
      edge.semanticScope = scope.str();
    }

    for (auto [edgeIndex, edge] : llvm::enumerate(edges)) {
      if (edge.target != target || edge.origin != "taskflow")
        continue;
      if (!annotatedEdges.contains(edgeIndex)) {
        auto sourceTask = tasksById.find(edge.source);
        error = "missing semantic incoming edge annotation for actual "
                "Taskflow edge from " +
                (sourceTask == tasksById.end()
                     ? std::to_string(edge.source)
                     : sourceTask->second.getTaskName().str());
        return false;
      }
    }
  }
  return true;
}

static bool sameLoopDomain(TaskflowTaskOp first, TaskflowTaskOp second) {
  FailureOr<SmallVector<std::pair<int64_t, int64_t>>> firstDomain =
      loopDomain(first);
  FailureOr<SmallVector<std::pair<int64_t, int64_t>>> secondDomain =
      loopDomain(second);
  return succeeded(firstDomain) && succeeded(secondDomain) &&
         *firstDomain == *secondDomain;
}

static bool canMoveFirstToSecond(TaskflowTaskOp first, TaskflowTaskOp second) {
  if (!first || !second || first == second ||
      first->hasAttr("amoeba.control_predecessors") ||
      second->hasAttr("amoeba.control_predecessors") ||
      isNamedControlPredecessor(first) || isNamedControlPredecessor(second) ||
      first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second))
    return false;
  llvm::SmallPtrSet<Operation *, 8> intervening;
  for (Operation *operation = first->getNextNode();
       operation && operation != second.getOperation();
       operation = operation->getNextNode()) {
    if (isa<TaskflowChannelOp>(operation))
      continue;
    auto crossed = dyn_cast<TaskflowTaskOp>(operation);
    if (!crossed)
      return false;
    if (crossed->hasAttr("amoeba.control_predecessors") ||
        hasConflictingOriginalMemoryEffects(first, crossed))
      return false;
    intervening.insert(operation);
  }
  for (Value result : first->getResults())
    for (OpOperand &use : result.getUses())
      if (intervening.contains(use.getOwner()))
        return false;
  return true;
}

static bool isExclusiveChannelPath(Value producerResult, Value consumerInput) {
  Value current = consumerInput;
  while (auto channel = current.getDefiningOp<TaskflowChannelOp>()) {
    if (!current.hasOneUse())
      return false;
    current = channel.getSource();
  }
  return current == producerResult && producerResult.hasOneUse();
}

static bool safeProducerConsumerForwarding(TaskflowTaskOp producer,
                                           TaskflowTaskOp consumer,
                                           Value producerResult,
                                           Value consumerInput) {
  if (!producer.getBody().hasOneBlock() || !consumer.getBody().hasOneBlock() ||
      producer.getOriginalWriteMemrefs().size() !=
          producer.getWillWrites().size() ||
      consumer.getOriginalReadMemrefs().size() !=
          consumer.getWillReads().size() ||
      consumer.getOriginalWriteMemrefs().size() !=
          consumer.getWillWrites().size())
    return false;
  auto result = llvm::find(producer.getDoneWrites(), producerResult);
  auto input = llvm::find(consumer.getWillReads(), consumerInput);
  if (result == producer.getDoneWrites().end() ||
      input == consumer.getWillReads().end())
    return false;
  unsigned resultIndex = result - producer.getDoneWrites().begin();
  unsigned consumerReadIndex = input - consumer.getWillReads().begin();
  auto producerYield =
      dyn_cast<TaskflowYieldOp>(producer.getBody().front().getTerminator());
  if (!producerYield || resultIndex >= producerYield.getDoneWrites().size())
    return false;
  auto producerArgument =
      dyn_cast<BlockArgument>(producerYield.getDoneWrites()[resultIndex]);
  unsigned firstWrite = producer.getWillReads().size();
  if (!producerArgument ||
      producerArgument.getOwner() != &producer.getBody().front() ||
      producerArgument.getArgNumber() < firstWrite ||
      producerArgument.getArgNumber() >=
          firstWrite + producer.getWillWrites().size())
    return false;
  unsigned producerWriteIndex = producerArgument.getArgNumber() - firstWrite;
  Value intermediateRoot =
      producer.getOriginalWriteMemrefs()[producerWriteIndex];
  if (consumer.getOriginalReadMemrefs()[consumerReadIndex] !=
          intermediateRoot ||
      !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
          intermediateRoot.getDefiningOp()))
    return false;
  for (OpOperand &use : intermediateRoot.getUses())
    if (use.getOwner() != producer.getOperation() &&
        use.getOwner() != consumer.getOperation() &&
        !isa<memref::DeallocOp>(use.getOwner()))
      return false;
  if (llvm::is_contained(producer.getOriginalReadMemrefs(), intermediateRoot) ||
      llvm::is_contained(consumer.getOriginalWriteMemrefs(), intermediateRoot))
    return false;
  for (auto [readIndex, root] :
       llvm::enumerate(consumer.getOriginalReadMemrefs()))
    if (readIndex != consumerReadIndex && root == intermediateRoot)
      return false;
  for (auto [writeIndex, root] :
       llvm::enumerate(producer.getOriginalWriteMemrefs())) {
    if (writeIndex == producerWriteIndex || root != intermediateRoot)
      continue;
    BlockArgument argument = producer.getBody().front().getArgument(
        producer.getWillReads().size() + writeIndex);
    if (!argument.use_empty())
      return false;
  }

  TaskflowHyperblockOp producerHyperblock;
  TaskflowHyperblockOp consumerHyperblock;
  for (Operation &operation : producer.getBody().front())
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (producerHyperblock)
        return false;
      producerHyperblock = hyperblock;
    }
  for (Operation &operation : consumer.getBody().front())
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (consumerHyperblock)
        return false;
      consumerHyperblock = hyperblock;
    }
  if (!producerHyperblock || !consumerHyperblock)
    return false;
  Block &producerKernel = producerHyperblock.getBody().front();
  Block &consumerKernel = consumerHyperblock.getBody().front();
  auto directPositions = [](ValueRange indices,
                            Block &body) -> FailureOr<SmallVector<unsigned>> {
    SmallVector<unsigned> positions;
    llvm::SmallDenseSet<unsigned, 4> unique;
    for (Value index : indices) {
      auto argument = dyn_cast<BlockArgument>(index);
      if (!argument || argument.getOwner() != &body ||
          !unique.insert(argument.getArgNumber()).second)
        return failure();
      positions.push_back(argument.getArgNumber());
    }
    return positions;
  };

  memref::StoreOp producerStore;
  for (OpOperand &use : producerArgument.getUses()) {
    if (isa<TaskflowYieldOp>(use.getOwner()))
      continue;
    auto store = dyn_cast<memref::StoreOp>(use.getOwner());
    if (!store || store->getBlock() != &producerKernel || producerStore)
      return false;
    producerStore = store;
  }
  if (!producerStore)
    return false;
  FailureOr<SmallVector<unsigned>> producerIndices =
      directPositions(producerStore.getIndices(), producerKernel);
  if (failed(producerIndices))
    return false;

  BlockArgument consumerArgument =
      consumer.getBody().front().getArgument(consumerReadIndex);
  bool sawLoad = false;
  for (OpOperand &use : consumerArgument.getUses()) {
    auto load = dyn_cast<memref::LoadOp>(use.getOwner());
    if (!load || load->getBlock() != &consumerKernel)
      return false;
    FailureOr<SmallVector<unsigned>> consumerIndices =
        directPositions(load.getIndices(), consumerKernel);
    if (failed(consumerIndices) || *consumerIndices != *producerIndices)
      return false;
    sawLoad = true;
  }
  return sawLoad;
}

static bool directProducerConsumer(TaskflowTaskOp first,
                                   TaskflowTaskOp second) {
  if (!canMoveFirstToSecond(first, second) || !sameLoopDomain(first, second))
    return false;
  unsigned edges = 0;
  Value producerIntermediate;
  Value consumerIntermediate;
  for (Value input : second.getWillReads()) {
    Value source = peelChannel(input);
    if (llvm::is_contained(first.getDoneWrites(), source) &&
        isExclusiveChannelPath(source, input)) {
      ++edges;
      producerIntermediate = source;
      consumerIntermediate = input;
    }
  }
  if (edges != 1)
    return false;
  for (Value result : first.getDoneWrites())
    if (result != producerIntermediate && !result.use_empty())
      return false;
  for (Value result : first.getValueOutputs())
    if (!result.use_empty())
      return false;
  return safeProducerConsumerForwarding(first, second, producerIntermediate,
                                        consumerIntermediate);
}

static bool independentSiblings(TaskflowTaskOp first, TaskflowTaskOp second) {
  if (!canMoveFirstToSecond(first, second) || !sameLoopDomain(first, second))
    return false;
  for (Value operand : second->getOperands())
    if (peelChannel(operand).getDefiningOp<TaskflowTaskOp>() == first)
      return false;
  for (Value operand : first->getOperands())
    if (peelChannel(operand).getDefiningOp<TaskflowTaskOp>() == second)
      return false;
  llvm::SmallPtrSet<Value, 8> firstInputs;
  for (Value input : first.getWillReads())
    firstInputs.insert(peelChannel(input));
  for (Value input : first.getValueInputs())
    firstInputs.insert(peelChannel(input));
  for (Value input : second.getWillReads())
    if (firstInputs.contains(peelChannel(input)))
      return true;
  for (Value input : second.getValueInputs())
    if (firstInputs.contains(peelChannel(input)))
      return true;
  return false;
}

struct DirectRawConnection {
  unsigned producerWriteIndex = 0;
  unsigned consumerReadIndex = 0;
};

static std::optional<DirectRawConnection>
directRawProducerConsumer(TaskflowTaskOp producer, TaskflowTaskOp consumer) {
  if (!producer || !consumer || producer == consumer ||
      !producer.getBody().hasOneBlock() || !consumer.getBody().hasOneBlock() ||
      producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) ||
      producer.getOriginalWriteMemrefs().size() !=
          producer.getWillWrites().size() ||
      consumer.getOriginalReadMemrefs().size() !=
          consumer.getWillReads().size())
    return std::nullopt;
  std::optional<DirectRawConnection> connection;
  for (auto [inputIndex, input] : llvm::enumerate(consumer.getWillReads())) {
    Value source = peelChannel(input);
    auto result = llvm::find(producer.getDoneWrites(), source);
    if (result == producer.getDoneWrites().end())
      continue;
    unsigned resultIndex = result - producer.getDoneWrites().begin();
    auto taskYield =
        dyn_cast<TaskflowYieldOp>(producer.getBody().front().getTerminator());
    if (!taskYield || resultIndex >= taskYield.getDoneWrites().size())
      return std::nullopt;
    auto argument =
        dyn_cast<BlockArgument>(taskYield.getDoneWrites()[resultIndex]);
    unsigned firstWrite = producer.getWillReads().size();
    if (!argument || argument.getOwner() != &producer.getBody().front() ||
        argument.getArgNumber() < firstWrite)
      return std::nullopt;
    unsigned writeIndex = argument.getArgNumber() - firstWrite;
    if (writeIndex >= producer.getOriginalWriteMemrefs().size() ||
        producer.getOriginalWriteMemrefs()[writeIndex] !=
            consumer.getOriginalReadMemrefs()[inputIndex])
      return std::nullopt;
    if (connection)
      return std::nullopt;
    connection =
        DirectRawConnection{writeIndex, static_cast<unsigned>(inputIndex)};
  }
  return connection;
}

struct CanonicalTask {
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

enum class CotilingAxisCompatibility : uint8_t {
  Compatible,
  Incompatible,
  Unknown,
};

static FailureOr<std::optional<unsigned>>
indexCounterAxis(Value value, Block &hyperblockBody) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &hyperblockBody)
      return failure();
    return std::optional<unsigned>(argument.getArgNumber());
  }
  if (constantIndex(value))
    return std::optional<unsigned>();
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    FailureOr<std::optional<unsigned>> lhs =
        indexCounterAxis(add.getLhs(), hyperblockBody);
    FailureOr<std::optional<unsigned>> rhs =
        indexCounterAxis(add.getRhs(), hyperblockBody);
    if (failed(lhs) || failed(rhs) || (lhs->has_value() && rhs->has_value()))
      return failure();
    return lhs->has_value() ? *lhs : *rhs;
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    FailureOr<std::optional<unsigned>> lhs =
        indexCounterAxis(sub.getLhs(), hyperblockBody);
    FailureOr<std::optional<unsigned>> rhs =
        indexCounterAxis(sub.getRhs(), hyperblockBody);
    if (failed(lhs) || failed(rhs) || rhs->has_value())
      return failure();
    return *lhs;
  }
  return failure();
}

static FailureOr<unsigned> producerStorageDimension(TaskflowTaskOp task,
                                                    CanonicalTask &canonical,
                                                    unsigned writeIndex,
                                                    unsigned counterAxis) {
  if (writeIndex >= task.getWillWrites().size())
    return failure();
  Block &taskBody = task.getBody().front();
  BlockArgument output =
      taskBody.getArgument(task.getWillReads().size() + writeIndex);
  std::optional<unsigned> storageDimension;
  bool valid = true;
  bool sawStore = false;
  task.walk([&](memref::StoreOp store) {
    if (!valid || store.getMemRef() != output)
      return;
    if (store->getBlock() != &canonical.hyperblock.getBody().front()) {
      valid = false;
      return;
    }
    sawStore = true;
    std::optional<unsigned> useDimension;
    for (auto [dimension, index] : llvm::enumerate(store.getIndices())) {
      auto argument = dyn_cast<BlockArgument>(index);
      if (!argument ||
          argument.getOwner() != &canonical.hyperblock.getBody().front()) {
        valid = false;
        return;
      }
      if (argument.getArgNumber() == counterAxis) {
        if (useDimension) {
          valid = false;
          return;
        }
        useDimension = dimension;
      }
    }
    if (!valid || !useDimension ||
        (storageDimension && *storageDimension != *useDimension)) {
      valid = false;
      return;
    }
    storageDimension = useDimension;
  });
  if (!valid || !sawStore || !storageDimension)
    return failure();
  return *storageDimension;
}

static FailureOr<std::optional<unsigned>>
consumerStorageDimension(TaskflowTaskOp task, CanonicalTask &canonical,
                         unsigned readIndex, unsigned counterAxis) {
  if (readIndex >= task.getWillReads().size())
    return failure();
  BlockArgument input = task.getBody().front().getArgument(readIndex);
  std::optional<unsigned> storageDimension;
  bool sawLoad = false;
  bool valid = true;
  bool selectedAxisMissing = false;
  task.walk([&](memref::LoadOp load) {
    if (!valid || load.getMemRef() != input)
      return;
    if (load->getBlock() != &canonical.hyperblock.getBody().front()) {
      valid = false;
      return;
    }
    sawLoad = true;
    std::optional<unsigned> useDimension;
    for (auto [dimension, index] : llvm::enumerate(load.getIndices())) {
      FailureOr<std::optional<unsigned>> projected =
          indexCounterAxis(index, canonical.hyperblock.getBody().front());
      if (failed(projected)) {
        valid = false;
        return;
      }
      if (*projected && **projected == counterAxis) {
        if (useDimension) {
          valid = false;
          return;
        }
        useDimension = dimension;
      }
    }
    if (!useDimension) {
      selectedAxisMissing = true;
      return;
    }
    if (storageDimension && *storageDimension != *useDimension) {
      valid = false;
      return;
    }
    storageDimension = useDimension;
  });
  if (!valid || !sawLoad)
    return failure();
  if (selectedAxisMissing)
    return std::optional<unsigned>();
  return storageDimension;
}

static CotilingAxisCompatibility cotilingAxisCompatibility(
    TaskflowTaskOp producer, CanonicalTask &producerCanonical,
    unsigned producerAxis, TaskflowTaskOp consumer,
    CanonicalTask &consumerCanonical, unsigned consumerAxis,
    DirectRawConnection connection) {
  FailureOr<unsigned> producerDimension = producerStorageDimension(
      producer, producerCanonical, connection.producerWriteIndex, producerAxis);
  FailureOr<std::optional<unsigned>> consumerDimension =
      consumerStorageDimension(consumer, consumerCanonical,
                               connection.consumerReadIndex, consumerAxis);
  if (failed(producerDimension) || failed(consumerDimension))
    return CotilingAxisCompatibility::Unknown;
  if (!*consumerDimension || *producerDimension != **consumerDimension)
    return CotilingAxisCompatibility::Incompatible;
  return CotilingAxisCompatibility::Compatible;
}

static FailureOr<CanonicalTask> canonicalTask(TaskflowTaskOp task) {
  if (!task.getBody().hasOneBlock())
    return failure();
  CanonicalTask canonical;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      canonical.counters.push_back(counter);
      continue;
    }
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (canonical.hyperblock)
        return failure();
      canonical.hyperblock = hyperblock;
      continue;
    }
    if (!isa<TaskflowYieldOp, arith::ConstantOp>(&operation))
      return failure();
  }
  if (!canonical.hyperblock || canonical.counters.empty() ||
      !canonical.hyperblock.getIterArgs().empty() ||
      canonical.hyperblock.getNumResults() != 0 ||
      canonical.hyperblock.getIndices().size() != canonical.counters.size())
    return failure();
  for (auto [index, counter] : llvm::enumerate(canonical.counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower ||
        (index == 0 && counter.getParentIndex()) ||
        (index != 0 && counter.getParentIndex() !=
                           canonical.counters[index - 1].getCounterIndex()) ||
        canonical.hyperblock.getIndices()[index] != counter.getCounterIndex())
      return failure();
    canonical.lower.push_back(*lower);
    canonical.upper.push_back(*upper);
  }
  return canonical;
}

static bool isStorableScalar(Type type) { return type.isIntOrIndexOrFloat(); }

static bool hasDirectIndices(ValueRange indices, Block &hyperblockBody) {
  return indices.size() == hyperblockBody.getNumArguments() &&
         llvm::equal(indices, hyperblockBody.getArguments());
}

static FailureOr<SmallVector<SmallVector<unsigned>>>
legalFissionPartitions(TaskflowTaskOp task, CanonicalTask &canonical,
                       uint64_t maxActions, std::string &error) {
  SmallVector<SmallVector<unsigned>> partitions;
  if (isNamedControlPredecessor(task) || !task.getDoneReads().empty() ||
      task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != 1)
    return partitions;
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "task_name" && name != "operandSegmentSizes" &&
        name != "resultSegmentSizes" && name != "operand_segment_sizes" &&
        name != "result_segment_sizes" &&
        !name.starts_with("amoeba.fission.") &&
        !name.starts_with("amoeba.fusion.") &&
        !name.starts_with("amoeba.tiling."))
      return partitions;
  }
  if (llvm::any_of(canonical.lower, [](int64_t lower) { return lower != 0; }))
    return partitions;
  Block &taskBody = task.getBody().front();
  size_t expectedArguments = task.getWillReads().size() +
                             task.getWillWrites().size() +
                             task.getValueInputs().size();
  auto taskYield = dyn_cast<TaskflowYieldOp>(taskBody.getTerminator());
  if (taskBody.getNumArguments() != expectedArguments || !taskYield ||
      !taskYield.getDoneReads().empty() ||
      taskYield.getDoneWrites().size() != 1 ||
      taskYield.getDoneWrites().front() !=
          taskBody.getArgument(task.getWillReads().size()) ||
      !taskYield.getValueResults().empty())
    return partitions;

  Block &body = canonical.hyperblock.getBody().front();
  SmallVector<Operation *> nodes;
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumRegions() != 0)
      return partitions;
    if (!isa<memref::LoadOp, memref::StoreOp>(&operation) &&
        (!isPure(&operation) || !isMemoryEffectFree(&operation)))
      return partitions;
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      bool knownRead = false;
      for (size_t index = 0; index < task.getWillReads().size(); ++index)
        knownRead |= load.getMemRef() == taskBody.getArgument(index);
      if (!knownRead || !hasDirectIndices(load.getIndices(), body))
        return partitions;
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation))
      if (store.getMemRef() !=
              taskBody.getArgument(task.getWillReads().size()) ||
          !hasDirectIndices(store.getIndices(), body))
        return partitions;
    nodes.push_back(&operation);
  }
  if (nodes.size() < 2)
    return partitions;
  DenseSet<Value> arguments(body.getArguments().begin(),
                            body.getArguments().end());
  DenseMap<Value, unsigned> producers;
  SmallVector<SmallVector<unsigned>> predecessors(nodes.size());
  SmallVector<bool> prefixAvailable(nodes.size(), true);
  for (auto [index, node] : llvm::enumerate(nodes)) {
    for (Value operand : node->getOperands()) {
      if (arguments.contains(operand))
        continue;
      if (isa<BlockArgument>(operand)) {
        if (!isKnownFissionBlockArgument(operand, task, body))
          return partitions;
        prefixAvailable[index] &=
            isAvailableInFissionPrefix(operand, task, body);
        continue;
      }
      auto producer = producers.find(operand);
      if (producer == producers.end() || producer->second >= index)
        return partitions;
      predecessors[index].push_back(producer->second);
    }
    llvm::sort(predecessors[index]);
    predecessors[index].erase(
        std::unique(predecessors[index].begin(), predecessors[index].end()),
        predecessors[index].end());
    for (Value result : node->getResults())
      producers[result] = index;
  }

  SmallVector<bool> selected(nodes.size(), false);
  SmallVector<unsigned> leftNodes;
  bool overflow = false;
  std::function<void(unsigned)> enumerate = [&](unsigned nodeIndex) {
    if (overflow)
      return;
    if (nodeIndex != nodes.size()) {
      enumerate(nodeIndex + 1);
      if (prefixAvailable[nodeIndex] &&
          !isa<memref::StoreOp>(nodes[nodeIndex]) &&
          llvm::all_of(predecessors[nodeIndex], [&](unsigned predecessor) {
            return selected[predecessor];
          })) {
        selected[nodeIndex] = true;
        leftNodes.push_back(nodeIndex);
        enumerate(nodeIndex + 1);
        leftNodes.pop_back();
        selected[nodeIndex] = false;
      }
      return;
    }
    if (leftNodes.empty() || leftNodes.size() == nodes.size())
      return;
    bool crosses = false;
    for (unsigned index = 0; index < nodes.size(); ++index) {
      if (selected[index])
        continue;
      for (Value operand : nodes[index]->getOperands()) {
        auto found = producers.find(operand);
        if (found != producers.end() && selected[found->second]) {
          if (!isStorableScalar(operand.getType()))
            return;
          crosses = true;
        }
      }
    }
    if (!crosses)
      return;
    if (partitions.size() == maxActions) {
      overflow = true;
      return;
    }
    partitions.push_back(leftNodes);
  };
  enumerate(0);
  if (overflow) {
    error = "legal fission cut-set count exceeds max-fission-actions-per-task";
    return failure();
  }
  llvm::sort(partitions);
  return partitions;
}

static bool isNoAliasFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

static bool provesDistinctStorage(Value lhs, Value rhs) {
  if (lhs == rhs)
    return false;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  bool lhsAllocation =
      lhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(lhsDefinition);
  bool rhsAllocation =
      rhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(rhsDefinition);
  if (lhsAllocation && rhsAllocation)
    return lhsDefinition != rhsDefinition;
  if (lhsAllocation && isNoAliasFunctionArgument(rhs))
    return true;
  if (rhsAllocation && isNoAliasFunctionArgument(lhs))
    return true;
  return isNoAliasFunctionArgument(lhs) && isNoAliasFunctionArgument(rhs);
}

static bool tilingContract(TaskflowTaskOp task, CanonicalTask &canonical,
                           size_t axis, std::string *reason = nullptr) {
  auto reject = [&](StringRef message) {
    if (reason)
      *reason = message.str();
    return false;
  };
  if (isNamedControlPredecessor(task))
    return reject("task_has_control_predecessor");
  if (task.getWillWrites().empty())
    return reject("no_writable_output");
  if (task.getWillWrites().size() != task.getDoneWrites().size() ||
      task.getWillWrites().size() != task.getOriginalWriteMemrefs().size())
    return reject("incomplete_write_provenance");
  if (!task.getDoneReads().empty())
    return reject("read_completion_result");
  if (!task.getValueOutputs().empty())
    return reject("value_output");
  if (task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return reject("incomplete_read_provenance");
  if (axis >= canonical.counters.size())
    return reject("axis_out_of_range");
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "operandSegmentSizes" && name != "resultSegmentSizes" &&
        name != "operand_segment_sizes" && name != "result_segment_sizes" &&
        name != "task_name" && !name.starts_with("amoeba.tiling.") &&
        !name.starts_with("amoeba.fusion.") &&
        !name.starts_with("amoeba.fission."))
      return reject("unsupported_task_attribute");
  }
  Block &taskBody = task.getBody().front();
  Block &kernelBody = canonical.hyperblock.getBody().front();
  auto yield = dyn_cast<TaskflowYieldOp>(taskBody.getTerminator());
  if (!yield || yield.getDoneWrites().size() != task.getWillWrites().size())
    return reject("noncanonical_write_yield");
  unsigned firstWriteArgument = task.getWillReads().size();
  SmallVector<MemRefType> outputTypes(task.getWillWrites().size());
  SmallVector<Value> writeArguments(task.getWillWrites().size());
  SmallVector<bool> mappedWrites(task.getWillWrites().size(), false);
  for (Value yielded : yield.getDoneWrites()) {
    auto argument = dyn_cast<BlockArgument>(yielded);
    if (!argument || argument.getOwner() != &taskBody ||
        argument.getArgNumber() < firstWriteArgument ||
        argument.getArgNumber() >=
            firstWriteArgument + task.getWillWrites().size())
      return reject("noncanonical_write_yield");
    unsigned writeIndex = argument.getArgNumber() - firstWriteArgument;
    auto type = dyn_cast<MemRefType>(argument.getType());
    if (mappedWrites[writeIndex] || !type || !type.hasStaticShape() ||
        !type.getLayout().isIdentity() ||
        task.getOriginalWriteMemrefs()[writeIndex].getType() != type)
      return reject("unsupported_output_layout_or_provenance");
    mappedWrites[writeIndex] = true;
    outputTypes[writeIndex] = type;
    writeArguments[writeIndex] = argument;
  }
  for (auto [lhsIndex, lhs] : llvm::enumerate(task.getOriginalWriteMemrefs()))
    for (Value rhs : task.getOriginalWriteMemrefs().drop_front(lhsIndex + 1))
      if (!provesDistinctStorage(lhs, rhs))
        return reject("ambiguous_output_alias");

  SmallVector<std::optional<unsigned>> readOutput(task.getWillReads().size());
  llvm::SmallPtrSet<Value, 8> readOnlyInputs;
  for (auto [readIndex, root] :
       llvm::enumerate(task.getOriginalReadMemrefs())) {
    Value argument = taskBody.getArgument(readIndex);
    if (argument.getType() != root.getType())
      return reject("mismatched_input_provenance");
    for (auto [writeIndex, outputRoot] :
         llvm::enumerate(task.getOriginalWriteMemrefs())) {
      if (root == outputRoot) {
        readOutput[readIndex] = writeIndex;
        break;
      }
      if (!provesDistinctStorage(root, outputRoot))
        return reject("ambiguous_input_output_alias");
    }
    if (!readOutput[readIndex])
      readOnlyInputs.insert(argument);
  }

  SmallVector<SmallVector<Value>> expectedOutputIndices(outputTypes.size());
  SmallVector<SmallVector<unsigned>> outputCounterAxes(outputTypes.size());
  SmallVector<bool> sawStore(outputTypes.size(), false);
  bool valid = true;
  auto checkOutputIndices = [&](unsigned outputIndex, ValueRange indices) {
    MemRefType outputType = outputTypes[outputIndex];
    if (!valid || indices.size() != static_cast<size_t>(outputType.getRank())) {
      valid = false;
      return;
    }
    if (expectedOutputIndices[outputIndex].empty()) {
      llvm::SmallDenseSet<unsigned, 4> uniqueAxes;
      for (Value index : indices) {
        auto found = llvm::find(kernelBody.getArguments(), index);
        if (found == kernelBody.getArguments().end()) {
          valid = false;
          return;
        }
        unsigned counterAxis =
            static_cast<unsigned>(found - kernelBody.getArguments().begin());
        if (!uniqueAxes.insert(counterAxis).second) {
          valid = false;
          return;
        }
        expectedOutputIndices[outputIndex].push_back(index);
        outputCounterAxes[outputIndex].push_back(counterAxis);
      }
      return;
    }
    if (!llvm::equal(indices, expectedOutputIndices[outputIndex]))
      valid = false;
  };
  task.walk([&](Operation *operation) {
    if (!valid || operation == task.getOperation() ||
        isa<TaskflowYieldOp, TaskflowCounterOp, TaskflowHyperblockOp,
            TaskflowHyperblockYieldOp, arith::ConstantOp>(operation))
      return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto readArgument = dyn_cast<BlockArgument>(load.getMemRef());
      if (readArgument && readArgument.getOwner() == &taskBody &&
          readArgument.getArgNumber() < readOutput.size() &&
          readOutput[readArgument.getArgNumber()]) {
        checkOutputIndices(*readOutput[readArgument.getArgNumber()],
                           load.getIndices());
        return;
      }
      auto outputArgument = llvm::find(writeArguments, load.getMemRef());
      if (outputArgument != writeArguments.end()) {
        checkOutputIndices(outputArgument - writeArguments.begin(),
                           load.getIndices());
      } else if (!readOnlyInputs.contains(load.getMemRef())) {
        valid = false;
      }
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto outputArgument = llvm::find(writeArguments, store.getMemRef());
      if (outputArgument == writeArguments.end()) {
        valid = false;
        return;
      }
      unsigned outputIndex = outputArgument - writeArguments.begin();
      checkOutputIndices(outputIndex, store.getIndices());
      sawStore[outputIndex] = true;
      return;
    }
    valid &= operation->getNumRegions() == 0 && isMemoryEffectFree(operation);
  });
  if (!valid)
    return reject("unsupported_or_nonrectangular_access");
  for (auto [outputIndex, outputType] : llvm::enumerate(outputTypes)) {
    if (!sawStore[outputIndex] ||
        !llvm::is_contained(outputCounterAxes[outputIndex], axis))
      return reject("selected_axis_not_in_every_output");
    for (auto [dimension, counterAxis] :
         llvm::enumerate(outputCounterAxes[outputIndex]))
      if (canonical.lower[counterAxis] < 0 ||
          canonical.upper[counterAxis] > outputType.getShape()[dimension])
        return reject("output_region_out_of_bounds");
  }
  return true;
}

static json::Object fusionAction(StringRef mode, StringRef first,
                                 StringRef second) {
  return json::Object{{"kind", "fusion"},
                      {"mode", mode},
                      {"first_task", first},
                      {"second_task", second}};
}

static json::Object cotilingAction(StringRef producer, StringRef consumer,
                                   int64_t producerAxis, int64_t consumerAxis,
                                   int64_t factor) {
  return json::Object{{"kind", "cotiling"},
                      {"producer_task", producer},
                      {"consumer_task", consumer},
                      {"producer_axis", producerAxis},
                      {"consumer_axis", consumerAxis},
                      {"factor", factor}};
}

static json::Array integerArray(ArrayRef<int64_t> values) {
  json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

static std::optional<json::Object> tileDescriptor(TaskflowTaskOp task) {
  auto parent = task->getAttrOfType<StringAttr>("amoeba.tiling.parent_task");
  if (!parent)
    return std::nullopt;
  auto axis = task->getAttrOfType<IntegerAttr>("amoeba.tiling.axis");
  auto factor = task->getAttrOfType<IntegerAttr>("amoeba.tiling.factor");
  auto part = task->getAttrOfType<IntegerAttr>("amoeba.tiling.part_index");
  auto step = task->getAttrOfType<IntegerAttr>("amoeba.tiling.step");
  auto original =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.original_range");
  auto derived =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.derived_range");
  if (!axis || !factor || !part || !step || !original || !derived)
    return std::nullopt;

  json::Array inputs;
  auto inputLowers =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionLowersAttr);
  auto inputUppers =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionUppersAttr);
  auto inputReasons =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionReasonsAttr);
  if (inputLowers && inputUppers && inputReasons &&
      inputLowers.size() == task.getWillReads().size() &&
      inputUppers.size() == inputLowers.size() &&
      inputReasons.size() == inputLowers.size()) {
    for (size_t index = 0; index < inputLowers.size(); ++index) {
      json::Object input{{"operand_index", static_cast<int64_t>(index)}};
      if (auto reason = dyn_cast<StringAttr>(inputReasons[index]))
        input["reason"] = reason.getValue().str();
      auto lower = dyn_cast<DenseI64ArrayAttr>(inputLowers[index]);
      auto upper = dyn_cast<DenseI64ArrayAttr>(inputUppers[index]);
      if (lower && upper) {
        input["scope"] = "tile_local";
        input["lower"] = integerArray(lower.asArrayRef());
        input["upper"] = integerArray(upper.asArrayRef());
      } else {
        input["scope"] = "tensor_wide";
      }
      inputs.push_back(std::move(input));
    }
  }

  json::Array outputs;
  auto outputAxes =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.output_axes");
  auto outputResults = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.tiling.output_result_indices");
  auto outputLowers =
      task->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
  auto outputUppers =
      task->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
  if (outputAxes && outputResults && outputLowers && outputUppers &&
      outputAxes.size() == static_cast<int64_t>(task.getWillWrites().size()) &&
      outputResults.size() == outputAxes.size() &&
      outputLowers.size() == task.getWillWrites().size() &&
      outputUppers.size() == outputLowers.size()) {
    for (size_t index = 0; index < outputLowers.size(); ++index) {
      auto lower = dyn_cast<DenseI64ArrayAttr>(outputLowers[index]);
      auto upper = dyn_cast<DenseI64ArrayAttr>(outputUppers[index]);
      if (!lower || !upper)
        continue;
      outputs.push_back(
          json::Object{{"write_index", static_cast<int64_t>(index)},
                       {"result_index", outputResults[index]},
                       {"axis", outputAxes[index]},
                       {"lower", integerArray(lower.asArrayRef())},
                       {"upper", integerArray(upper.asArrayRef())}});
    }
  }
  return json::Object{{"source_task", parent.getValue()},
                      {"axis", axis.getInt()},
                      {"partition_count", factor.getInt()},
                      {"partition_index", part.getInt()},
                      {"step", step.getInt()},
                      {"original_range", integerArray(original.asArrayRef())},
                      {"derived_range", integerArray(derived.asArrayRef())},
                      {"inputs", std::move(inputs)},
                      {"outputs", std::move(outputs)}};
}

struct ExtractJointTaskGraphFactsPass
    : public PassWrapper<ExtractJointTaskGraphFactsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractJointTaskGraphFactsPass)

  ExtractJointTaskGraphFactsPass() = default;
  ExtractJointTaskGraphFactsPass(const ExtractJointTaskGraphFactsPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const override {
    return "extract-joint-task-graph-facts";
  }
  StringRef getDescription() const override {
    return "Extract graph facts and conservative legal rewrite actions";
  }
  Option<std::string> functionName{*this, "function",
                                   llvm::cl::desc("Taskflow function"),
                                   llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Graph facts JSON output"),
                                 llvm::cl::init("")};
  Option<int64_t> maxTileFactor{
      *this, "max-tile-factor",
      llvm::cl::desc("Largest generated task count per tiling action"),
      llvm::cl::init(8)};
  Option<int64_t> maxFissionActionsPerTask{
      *this, "max-fission-actions-per-task",
      llvm::cl::desc("Fail above this many legal DFG cut sets per task; zero "
                     "disables fission action generation"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected) || outputFile.empty() || maxTileFactor < 2 ||
        maxFissionActionsPerTask < 0) {
      module.emitError()
          << (error.empty() ? "graph facts require output, max-tile-factor >= "
                              "2, and a nonnegative fission action bound"
                            : error);
      return signalPassFailure();
    }
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    DenseMap<Operation *, unsigned> indices;
    for (auto [index, task] : llvm::enumerate(*metadata))
      indices[task.op.getOperation()] = index;
    FailureOr<SmallVector<EdgeFact, 0>> edges =
        collectEdges(*selected, indices, error);
    if (failed(edges)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    if (!validateSemanticIncomingEdges(*selected, indices, *edges, error)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    DenseMap<Value, unsigned> externalValueIds;
    unsigned nextExternalValueId = 0;
    for (BlockArgument argument : selected->getArguments())
      externalValueIds[argument] = nextExternalValueId++;
    for (Operation &operation : selected->getBody().front()) {
      if (isa<TaskflowTaskOp, TaskflowChannelOp, TaskflowJoinOp>(&operation))
        continue;
      for (Value result : operation.getResults())
        externalValueIds.try_emplace(result, nextExternalValueId++);
    }

    json::Array taskRecords;
    json::Array shapeNeeds;
    json::Array structuralNodes;
    json::Array actions;
    json::Array actionDiagnostics;
    int64_t consideredTiling = 0;
    int64_t acceptedTiling = 0;
    int64_t consideredFusion = 0;
    int64_t acceptedFusion = 0;
    int64_t consideredCotiling = 0;
    int64_t acceptedCotiling = 0;
    llvm::StringMap<int64_t> tilingRejections;
    llvm::StringMap<int64_t> fusionRejections;
    llvm::StringMap<int64_t> cotilingRejections;
    for (auto [index, task] : llvm::enumerate(*metadata)) {
      FailureOr<CanonicalTask> canonical = canonicalTask(task.op);
      if (failed(canonical)) {
        task.op.emitError("graph enumeration requires the canonical static "
                          "counter-chain plus one hyperblock phase");
        return signalPassFailure();
      }
      json::Array dimensions;
      FailureOr<SmallVector<std::pair<int64_t, int64_t>>> domain =
          loopDomain(task.op);
      if (succeeded(domain))
        for (auto [lower, upper] : *domain)
          dimensions.push_back(upper - lower);
      json::Object taskRecord{{"task_name", task.name},
                              {"body_sha256", task.bodySha256},
                              {"task_index", static_cast<int64_t>(index)},
                              {"cost_query_supported", true},
                              {"native_mapper_input_supported", true}};
      if (auto role =
              task.op->getAttrOfType<StringAttr>("amoeba.semantic.task_role"))
        taskRecord["semantic_role"] = role.getValue().str();
      if (auto kind =
              task.op->getAttrOfType<StringAttr>("amoeba.semantic.task_kind"))
        taskRecord["semantic_kind"] = kind.getValue().str();
      auto range =
          task.op->getAttrOfType<DenseI64ArrayAttr>("amoeba.semantic.K_range");
      if (!range)
        range = task.op->getAttrOfType<DenseI64ArrayAttr>(
            "amoeba.semantic.k_range");
      if (range)
        taskRecord["semantic_k_range"] = integerArray(range.asArrayRef());
      auto outputLowers =
          task.op->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
      auto outputUppers =
          task.op->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
      if (outputLowers && outputUppers &&
          outputLowers.size() == outputUppers.size()) {
        json::Array regions;
        for (auto [lowerAttr, upperAttr] :
             llvm::zip(outputLowers, outputUppers)) {
          auto lower = dyn_cast<DenseI64ArrayAttr>(lowerAttr);
          auto upper = dyn_cast<DenseI64ArrayAttr>(upperAttr);
          if (!lower || !upper)
            continue;
          regions.push_back(
              json::Object{{"lower", integerArray(lower.asArrayRef())},
                           {"upper", integerArray(upper.asArrayRef())}});
        }
        taskRecord["output_regions"] = std::move(regions);
      }
      if (std::optional<json::Object> descriptor = tileDescriptor(task.op))
        taskRecord["tile"] = std::move(*descriptor);
      taskRecords.push_back(std::move(taskRecord));
      shapeNeeds.push_back(json::Object{
          {"task_name", task.name}, {"static_dims", std::move(dimensions)}});
      json::Array inputBindings;
      auto addBinding = [&](StringRef segment, Value input) {
        SmallVector<std::pair<unsigned, std::string>> sources;
        DenseSet<Value> visited;
        sourceTaskIndices(input, indices, sources, visited);
        if (!sources.empty()) {
          inputBindings.push_back((segment + ":dependency").str());
          return;
        }
        Value root = peelChannel(input);
        auto found = externalValueIds.find(root);
        if (found == externalValueIds.end()) {
          inputBindings.push_back((segment + ":unknown").str());
          return;
        }
        inputBindings.push_back(
            (segment + ":external:" + Twine(found->second)).str());
      };
      for (Value input : task.op.getWillReads())
        addBinding("read", input);
      for (Value input : task.op.getWillWrites())
        addBinding("write", input);
      for (Value input : task.op.getValueInputs())
        addBinding("value", input);
      json::Object structuralNode{{"body_sha256", task.bodySha256},
                                  {"trip_count", task.tripCount},
                                  {"input_bindings", std::move(inputBindings)}};
      if (std::optional<json::Object> descriptor = tileDescriptor(task.op))
        structuralNode["tile"] = std::move(*descriptor);
      structuralNodes.push_back(std::move(structuralNode));

      if (maxFissionActionsPerTask > 0) {
        FailureOr<SmallVector<SmallVector<unsigned>>> fissionPartitions =
            legalFissionPartitions(task.op, *canonical,
                                   maxFissionActionsPerTask, error);
        if (failed(fissionPartitions)) {
          task.op.emitError() << error;
          return signalPassFailure();
        }
        for (ArrayRef<unsigned> leftNodes : *fissionPartitions) {
          json::Array ordinals;
          for (unsigned ordinal : leftNodes)
            ordinals.push_back(static_cast<int64_t>(ordinal));
          actions.push_back(json::Object{{"kind", "fission"},
                                         {"task_name", task.name},
                                         {"left_nodes", std::move(ordinals)}});
        }
      }
      for (size_t axis = 0; axis < canonical->counters.size(); ++axis) {
        ++consideredTiling;
        std::string rejectionReason;
        bool accepted =
            tilingContract(task.op, *canonical, axis, &rejectionReason);
        if (accepted) {
          ++acceptedTiling;
          int64_t extent = canonical->upper[axis] - canonical->lower[axis];
          int64_t limit = std::min<int64_t>(maxTileFactor, extent);
          for (int64_t factor = 2; factor <= limit; ++factor)
            actions.push_back(json::Object{{"kind", "tiling"},
                                           {"task_name", task.name},
                                           {"axis", static_cast<int64_t>(axis)},
                                           {"factor", factor}});
        } else {
          ++tilingRejections[rejectionReason];
        }
      }
    }
    for (size_t firstIndex = 0; firstIndex < metadata->size(); ++firstIndex)
      for (size_t secondIndex = firstIndex + 1; secondIndex < metadata->size();
           ++secondIndex) {
        TaskflowTaskOp first = (*metadata)[firstIndex].op;
        TaskflowTaskOp second = (*metadata)[secondIndex].op;
        ++consideredFusion;
        bool producerConsumer = directProducerConsumer(first, second);
        if (producerConsumer) {
          ++acceptedFusion;
          actions.push_back(fusionAction("producer_consumer",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
        } else {
          ++fusionRejections["producer_consumer_contract_rejected"];
        }
        ++consideredFusion;
        bool siblings = independentSiblings(first, second);
        if (siblings) {
          ++acceptedFusion;
          actions.push_back(fusionAction("sibling",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
        } else {
          ++fusionRejections["sibling_contract_rejected"];
        }
        std::optional<DirectRawConnection> rawConnection =
            directRawProducerConsumer(first, second);
        FailureOr<CanonicalTask> firstCanonical = canonicalTask(first);
        FailureOr<CanonicalTask> secondCanonical = canonicalTask(second);
        // Equal factors do not require equal loop bounds. Interior stencil
        // domains intentionally differ from their producers, and layout
        // changes can map a producer dimension to a differently numbered
        // consumer counter. After both rewrites, the typed-edge builder proves
        // exact read/write intersections or conservatively expands the join.
        size_t producerAxes =
            succeeded(firstCanonical) ? firstCanonical->counters.size() : 0;
        size_t consumerAxes =
            succeeded(secondCanonical) ? secondCanonical->counters.size() : 0;
        for (size_t producerAxis = 0; producerAxis < producerAxes;
             ++producerAxis)
          for (size_t consumerAxis = 0; consumerAxis < consumerAxes;
               ++consumerAxis) {
            ++consideredCotiling;
            std::string rejectionReason;
            if (!rawConnection) {
              rejectionReason = "no_direct_raw_connection";
            } else if (!tilingContract(first, *firstCanonical, producerAxis,
                                       &rejectionReason)) {
              rejectionReason = "producer_" + rejectionReason;
            } else if (!tilingContract(second, *secondCanonical, consumerAxis,
                                       &rejectionReason)) {
              rejectionReason = "consumer_" + rejectionReason;
            } else if (cotilingAxisCompatibility(
                           first, *firstCanonical, producerAxis, second,
                           *secondCanonical, consumerAxis, *rawConnection) ==
                       CotilingAxisCompatibility::Incompatible) {
              rejectionReason = "shared_storage_axis_mismatch";
            }
            if (!rejectionReason.empty()) {
              ++cotilingRejections[rejectionReason];
              continue;
            }
            ++acceptedCotiling;
            int64_t producerExtent = firstCanonical->upper[producerAxis] -
                                     firstCanonical->lower[producerAxis];
            int64_t consumerExtent = secondCanonical->upper[consumerAxis] -
                                     secondCanonical->lower[consumerAxis];
            int64_t limit = std::min({static_cast<int64_t>(maxTileFactor),
                                      producerExtent, consumerExtent});
            for (int64_t factor = 2; factor <= limit; ++factor)
              actions.push_back(cotilingAction(
                  (*metadata)[firstIndex].name, (*metadata)[secondIndex].name,
                  producerAxis, consumerAxis, factor));
          }
      }

    auto addActionSummary = [&](StringRef kind, int64_t considered,
                                int64_t accepted,
                                const llvm::StringMap<int64_t> &rejections) {
      SmallVector<StringRef> reasons;
      for (const auto &entry : rejections)
        reasons.push_back(entry.getKey());
      llvm::sort(reasons);
      json::Array reasonRecords;
      for (StringRef reason : reasons)
        reasonRecords.push_back(json::Object{
            {"reason", reason}, {"count", rejections.lookup(reason)}});
      StringRef summary = reasons.empty()       ? "none"
                          : reasons.size() == 1 ? reasons.front()
                                                : "multiple";
      actionDiagnostics.push_back(
          json::Object{{"kind", kind},
                       {"considered", considered},
                       {"accepted", accepted},
                       {"rejected", considered - accepted},
                       {"rejection_reason", summary},
                       {"rejection_reasons", std::move(reasonRecords)}});
    };
    addActionSummary("tiling", consideredTiling, acceptedTiling,
                     tilingRejections);
    addActionSummary("fusion", consideredFusion, acceptedFusion,
                     fusionRejections);
    addActionSummary("cotiling", consideredCotiling, acceptedCotiling,
                     cotilingRejections);

    json::Array edgeRecords;
    json::Array structuralEdges;
    for (const EdgeFact &edge : *edges) {
      json::Object edgeRecord{
          {"source_task", (*metadata)[edge.source].name},
          {"target_task", (*metadata)[edge.target].name},
          {"kind", edge.kind},
          {"producer_segment", edge.producerSegment},
          {"producer_index", edge.producerIndex},
          {"consumer_segment", edge.consumerSegment},
          {"consumer_index", edge.consumerIndex},
          {"origin", edge.origin},
          {"scope", edge.scope},
          {"semantic_role", edge.semanticRole.empty()
                                ? json::Value(nullptr)
                                : json::Value(edge.semanticRole)},
          {"semantic_scope", edge.semanticScope.empty()
                                 ? json::Value(nullptr)
                                 : json::Value(edge.semanticScope)},
          {"region_lower", nullptr},
          {"region_upper", nullptr}};
      json::Object structuralEdge{
          {"source", static_cast<int64_t>(edge.source)},
          {"target", static_cast<int64_t>(edge.target)},
          {"kind", edge.kind},
          {"producer_segment", edge.producerSegment},
          {"producer_index", edge.producerIndex},
          {"consumer_segment", edge.consumerSegment},
          {"consumer_index", edge.consumerIndex},
          {"origin", edge.origin},
          {"scope", edge.scope},
          {"semantic_role", edge.semanticRole.empty()
                                ? json::Value(nullptr)
                                : json::Value(edge.semanticRole)},
          {"semantic_scope", edge.semanticScope.empty()
                                 ? json::Value(nullptr)
                                 : json::Value(edge.semanticScope)},
          {"region_lower", nullptr},
          {"region_upper", nullptr}};
      if (edge.payloadBits) {
        edgeRecord["payload_bits"] = static_cast<int64_t>(*edge.payloadBits);
        edgeRecord["payload_bytes"] = static_cast<int64_t>(
            *edge.payloadBits / 8 + (*edge.payloadBits % 8 != 0));
        structuralEdge["payload_bits"] =
            static_cast<int64_t>(*edge.payloadBits);
      } else {
        edgeRecord["payload_bits"] = nullptr;
        edgeRecord["payload_bytes"] = nullptr;
        structuralEdge["payload_bits"] = nullptr;
      }
      if (!edge.regionLower.empty()) {
        edgeRecord["region_lower"] = integerArray(edge.regionLower);
        edgeRecord["region_upper"] = integerArray(edge.regionUpper);
        structuralEdge["region_lower"] = integerArray(edge.regionLower);
        structuralEdge["region_upper"] = integerArray(edge.regionUpper);
      }
      edgeRecords.push_back(std::move(edgeRecord));
      structuralEdges.push_back(std::move(structuralEdge));
    }
    json::Array completionJoins;
    int64_t joinIndex = 0;
    selected->walk([&](TaskflowJoinOp join) {
      completionJoins.push_back(json::Object{
          {"join_index", joinIndex++},
          {"axis", join.getAxis()},
          {"region_lower", integerArray(join.getRegionLower())},
          {"region_upper", integerArray(join.getRegionUpper())},
          {"producer_tiles", static_cast<int64_t>(join.getTileStates().size())},
          {"users",
           static_cast<int64_t>(std::distance(join.getJoined().use_begin(),
                                              join.getJoined().use_end()))}});
    });
    json::Object root{
        {"schema", kFactsSchema},
        {"function", selected->getSymName()},
        {"tasks", std::move(taskRecords)},
        {"dependencies", std::move(edgeRecords)},
        {"shape_needs", std::move(shapeNeeds)},
        {"completion_joins", std::move(completionJoins)},
        {"action_diagnostics", std::move(actionDiagnostics)},
        {"structural_key", json::Object{{"nodes", std::move(structuralNodes)},
                                        {"edges", std::move(structuralEdges)}}},
        {"legal_actions", std::move(actions)}};
    if (!writeAtomically(
            outputFile.getValue(),
            [&](llvm::raw_ostream &os) {
              os << json::Value(std::move(root)) << "\n";
              return true;
            },
            error)) {
      selected->emitError() << error;
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createExtractJointTaskGraphFactsPass() {
  return std::make_unique<ExtractJointTaskGraphFactsPass>();
}
} // namespace mlir::amoeba::neura
