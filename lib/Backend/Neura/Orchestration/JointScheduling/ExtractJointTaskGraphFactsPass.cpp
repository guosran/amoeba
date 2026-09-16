//===- ExtractJointTaskGraphFactsPass.cpp ---------------------*- C++ -*-===//
//
// Emits deterministic Taskflow graph facts and conservative legal rewrite
// actions. The graph-variant driver invokes this pass again after every
// rewrite; no legality decision is carried across a changed graph.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
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

  bool operator<(const EdgeFact &other) const {
    return std::tie(source, target, kind) <
           std::tie(other.source, other.target, other.kind);
  }
  bool operator==(const EdgeFact &other) const {
    return source == other.source && target == other.target &&
           kind == other.kind;
  }
};

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

static SmallVector<EdgeFact>
collectEdges(ArrayRef<TaskMetadata> tasks,
             const DenseMap<Operation *, unsigned> &indices) {
  SmallVector<EdgeFact> edges;
  for (auto [target, metadata] : llvm::enumerate(tasks)) {
    TaskflowTaskOp task = metadata.op;
    auto add = [&](Value input, StringRef requestedKind) {
      SmallVector<std::pair<unsigned, std::string>> sources;
      DenseSet<Value> visited;
      sourceTaskIndices(input, indices, sources, visited);
      for (auto &[source, sourceKind] : sources) {
        if (source == target)
          continue;
        std::string kind = requestedKind.str();
        if (requestedKind == "read")
          kind = sourceKind == "value" ? "value" : "raw";
        else if (requestedKind == "write")
          kind = sourceKind == "war" ? "war" : "waw";
        edges.push_back({source, static_cast<unsigned>(target), kind});
      }
    };
    for (Value input : task.getWillReads())
      add(input, "read");
    for (Value input : task.getWillWrites())
      add(input, "write");
    for (Value input : task.getValueInputs())
      add(input, "value");
  }
  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
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
      first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second))
    return false;
  llvm::SmallPtrSet<Operation *, 8> intervening;
  for (Operation *operation = first->getNextNode();
       operation && operation != second.getOperation();
       operation = operation->getNextNode()) {
    if (isa<TaskflowChannelOp>(operation))
      continue;
    if (!isa<TaskflowTaskOp>(operation))
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

static bool directProducerConsumer(TaskflowTaskOp first,
                                   TaskflowTaskOp second) {
  if (!canMoveFirstToSecond(first, second) || !sameLoopDomain(first, second))
    return false;
  unsigned edges = 0;
  Value producerIntermediate;
  for (Value input : second.getWillReads()) {
    Value source = peelChannel(input);
    if (llvm::is_contained(first.getDoneWrites(), source) &&
        isExclusiveChannelPath(source, input)) {
      ++edges;
      producerIntermediate = source;
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
  return true;
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

struct CanonicalTask {
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

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
  if (!task.getDoneReads().empty() || task.getWillWrites().size() != 1 ||
      task.getDoneWrites().size() != 1 || !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != 1)
    return partitions;
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "task_name" && name != "operandSegmentSizes" &&
        name != "resultSegmentSizes" && name != "operand_segment_sizes" &&
        name != "result_segment_sizes" &&
        !name.starts_with("amoeba.fission.") &&
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
  for (auto [index, node] : llvm::enumerate(nodes)) {
    for (Value operand : node->getOperands()) {
      if (arguments.contains(operand) || isa<BlockArgument>(operand))
        continue;
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
      if (!isa<memref::StoreOp>(nodes[nodeIndex]) &&
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

static bool tilingContract(TaskflowTaskOp task, CanonicalTask &canonical) {
  if (task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      !task.getDoneReads().empty() || !task.getValueOutputs().empty() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return false;
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "operandSegmentSizes" && name != "resultSegmentSizes" &&
        name != "operand_segment_sizes" && name != "result_segment_sizes" &&
        name != "task_name" && !name.starts_with("amoeba.tiling.") &&
        !name.starts_with("amoeba.fission."))
      return false;
  }
  SmallVector<Value> memories(task.getWillReads());
  memories.push_back(task.getWillWrites().front());
  for (Value value : memories) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        type.getRank() != static_cast<int64_t>(canonical.counters.size()))
      return false;
    for (size_t axis = 0; axis < canonical.counters.size(); ++axis)
      if (canonical.lower[axis] < 0 ||
          canonical.upper[axis] > type.getShape()[axis])
        return false;
  }
  SmallVector<Value> roots(task.getOriginalReadMemrefs());
  roots.push_back(task.getOriginalWriteMemrefs().front());
  for (size_t index = 0; index < roots.size(); ++index) {
    if (roots[index].getType() != memories[index].getType())
      return false;
    for (size_t other = index + 1; other < roots.size(); ++other)
      if (!provesDistinctStorage(roots[index], roots[other]))
        return false;
  }
  Block &taskBody = task.getBody().front();
  BlockArgument output = taskBody.getArgument(task.getWillReads().size());
  llvm::SmallPtrSet<Value, 8> reads;
  for (size_t index = 0; index < task.getWillReads().size(); ++index)
    reads.insert(taskBody.getArgument(index));
  bool valid = true;
  bool sawStore = false;
  task.walk([&](Operation *operation) {
    if (!valid || operation == task.getOperation() ||
        isa<TaskflowYieldOp, TaskflowCounterOp, TaskflowHyperblockOp,
            TaskflowHyperblockYieldOp, arith::ConstantOp>(operation))
      return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      valid &=
          reads.contains(load.getMemRef()) &&
          llvm::equal(load.getIndices(),
                      canonical.hyperblock.getBody().front().getArguments());
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      valid &=
          store.getMemRef() == output &&
          llvm::equal(store.getIndices(),
                      canonical.hyperblock.getBody().front().getArguments());
      sawStore = true;
      return;
    }
    valid &= operation->getNumRegions() == 0 && isMemoryEffectFree(operation);
  });
  return valid && sawStore;
}

static json::Object fusionAction(StringRef mode, StringRef first,
                                 StringRef second) {
  return json::Object{{"kind", "fusion"},
                      {"mode", mode},
                      {"first_task", first},
                      {"second_task", second}};
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
      llvm::cl::desc("Fail above this many legal DFG cut sets per task"),
      llvm::cl::init(256)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected) || outputFile.empty() || maxTileFactor < 2 ||
        maxFissionActionsPerTask <= 0) {
      module.emitError()
          << (error.empty() ? "graph facts require output, max-tile-factor >= "
                              "2, and a positive fission action bound"
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
    for (TaskMetadata task : *metadata)
      if (task.op->hasAttr("amoeba.control_predecessors")) {
        task.op.getOperation()->emitError(
            "graph facts do not yet encode explicit control predecessors");
        return signalPassFailure();
      }
    SmallVector<EdgeFact> edges = collectEdges(*metadata, indices);
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
      taskRecords.push_back(
          json::Object{{"task_name", task.name},
                       {"body_sha256", task.bodySha256},
                       {"task_index", static_cast<int64_t>(index)}});
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
      structuralNodes.push_back(
          json::Object{{"body_sha256", task.bodySha256},
                       {"input_bindings", std::move(inputBindings)}});

      FailureOr<SmallVector<SmallVector<unsigned>>> fissionPartitions =
          legalFissionPartitions(task.op, *canonical, maxFissionActionsPerTask,
                                 error);
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
      if (tilingContract(task.op, *canonical)) {
        for (size_t axis = 0; axis < canonical->counters.size(); ++axis) {
          int64_t extent = canonical->upper[axis] - canonical->lower[axis];
          int64_t limit = std::min<int64_t>(maxTileFactor, extent);
          for (int64_t factor = 2; factor <= limit; ++factor)
            actions.push_back(json::Object{{"kind", "tiling"},
                                           {"task_name", task.name},
                                           {"axis", static_cast<int64_t>(axis)},
                                           {"factor", factor}});
        }
      }
    }
    for (size_t firstIndex = 0; firstIndex < metadata->size(); ++firstIndex)
      for (size_t secondIndex = firstIndex + 1; secondIndex < metadata->size();
           ++secondIndex) {
        TaskflowTaskOp first = (*metadata)[firstIndex].op;
        TaskflowTaskOp second = (*metadata)[secondIndex].op;
        if (directProducerConsumer(first, second))
          actions.push_back(fusionAction("producer_consumer",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
        if (independentSiblings(first, second))
          actions.push_back(fusionAction("sibling",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
      }

    json::Array edgeRecords;
    json::Array structuralEdges;
    for (const EdgeFact &edge : edges) {
      edgeRecords.push_back(
          json::Object{{"source_task", (*metadata)[edge.source].name},
                       {"target_task", (*metadata)[edge.target].name},
                       {"kind", edge.kind}});
      structuralEdges.push_back(
          json::Object{{"source", static_cast<int64_t>(edge.source)},
                       {"target", static_cast<int64_t>(edge.target)},
                       {"kind", edge.kind}});
    }
    json::Object root{
        {"schema", kFactsSchema},
        {"function", selected->getSymName()},
        {"tasks", std::move(taskRecords)},
        {"dependencies", std::move(edgeRecords)},
        {"shape_needs", std::move(shapeNeeds)},
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
