//===- MaterializeJointSemanticRewritePass.cpp ----------------*- C++ -*-===//
//
// Hardware independent, source level rewrites for the ORBIT semantic task
// graph.  This file deliberately owns only transformations which can be
// proved from Taskflow IR.  It does not inspect an architecture, invoke a
// mapper, or attach a placement decision.
//
// The M/N tilers are intentionally independent passes.  This is useful to
// callers which enumerate a rewrite DAG: after every pass the next pass sees
// a fresh SSA graph and redoes its legality checks.  The sequential/parallel
// K entry points are also independent and fail closed until the input uses a
// task level K state chain (the canonical fixture represents K as an inner
// scf.for and therefore cannot be split without introducing a new state
// protocol).
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

using mlir::amoeba::neura::joint_scheduling::kTilingInputRegionLowersAttr;
using mlir::amoeba::neura::joint_scheduling::kTilingInputRegionReasonsAttr;
using mlir::amoeba::neura::joint_scheduling::kTilingInputRegionUppersAttr;
using mlir::amoeba::neura::joint_scheduling::kTilingOutputRegionLowersAttr;
using mlir::amoeba::neura::joint_scheduling::kTilingOutputRegionUppersAttr;

namespace {

constexpr StringLiteral kSemanticRoleAttr = "amoeba.semantic.task_role";
constexpr StringLiteral kSemanticKindAttr = "amoeba.semantic.task_kind";
constexpr StringLiteral kSemanticKRangeAttr = "amoeba.semantic.K_range";
constexpr StringLiteral kTiledAxesAttr = "amoeba.semantic.tiled_axes";
constexpr StringLiteral kTileAxisAttr = "amoeba.semantic.tile_axis";
constexpr StringLiteral kTileSizeAttr = "amoeba.semantic.tile_size";
constexpr StringLiteral kTileRangeAttr = "amoeba.semantic.tile_range";
constexpr StringLiteral kFusionModeAttr = "amoeba.semantic.fusion_mode";
constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";
constexpr StringLiteral kRewriteSchemaAttr = "amoeba.semantic.rewrite_schema";
constexpr StringLiteral kRewriteSchema = "orbit-joint-semantic-rewrite-v1";

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError(message);
  return failure();
}

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  return std::nullopt;
}

static bool isNoAliasArgument(Value value) {
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
  if (lhsAllocation && isNoAliasArgument(rhs))
    return true;
  if (rhsAllocation && isNoAliasArgument(lhs))
    return true;
  return isNoAliasArgument(lhs) && isNoAliasArgument(rhs);
}

static SmallVector<TaskflowCounterOp> counterChain(TaskflowTaskOp task) {
  SmallVector<TaskflowCounterOp> result;
  if (!task || !task.getBody().hasOneBlock())
    return result;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      result.push_back(counter);
  return result;
}

static TaskflowHyperblockOp findHyperblock(TaskflowTaskOp task) {
  TaskflowHyperblockOp result;
  if (!task || !task.getBody().hasOneBlock())
    return result;
  for (Operation &operation : task.getBody().front()) {
    auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation);
    if (!hyperblock)
      continue;
    if (result)
      return TaskflowHyperblockOp();
    result = hyperblock;
  }
  return result;
}

static bool sameCounterDomain(ArrayRef<TaskflowCounterOp> lhs,
                              ArrayRef<TaskflowCounterOp> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t index = 0; index < lhs.size(); ++index) {
    TaskflowCounterOp left = lhs[index];
    TaskflowCounterOp right = rhs[index];
    auto leftLower = constantIndex(left.getLowerBound());
    auto rightLower = constantIndex(right.getLowerBound());
    auto leftUpper = constantIndex(left.getUpperBound());
    auto rightUpper = constantIndex(right.getUpperBound());
    auto leftStep = constantIndex(left.getStep());
    auto rightStep = constantIndex(right.getStep());
    if (!leftLower || !rightLower || !leftUpper || !rightUpper || !leftStep ||
        !rightStep || *leftLower != *rightLower || *leftUpper != *rightUpper ||
        *leftStep != *rightStep)
      return false;
  }
  return true;
}

static bool sameLoopStructure(TaskflowTaskOp lhs, TaskflowTaskOp rhs) {
  auto leftCounters = counterChain(lhs);
  auto rightCounters = counterChain(rhs);
  if (!leftCounters.empty() || !rightCounters.empty())
    return !leftCounters.empty() &&
           sameCounterDomain(leftCounters, rightCounters);
  auto leftHyperblock = findHyperblock(lhs);
  auto rightHyperblock = findHyperblock(rhs);
  return leftHyperblock && rightHyperblock &&
         leftHyperblock.getIndices().size() ==
             rightHyperblock.getIndices().size() &&
         leftHyperblock.getIterArgs().empty() &&
         rightHyperblock.getIterArgs().empty();
}

static Value peelChannels(Value value) {
  while (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    value = channel.getSource();
  return value;
}

static bool exclusiveChannelPath(Value producerResult, Value consumerInput,
                                 SmallVectorImpl<TaskflowChannelOp> &channels) {
  Value current = consumerInput;
  while (auto channel = current.getDefiningOp<TaskflowChannelOp>()) {
    if (!current.hasOneUse())
      return false;
    channels.push_back(channel);
    current = channel.getSource();
  }
  return current == producerResult && producerResult.hasOneUse();
}

static bool canMoveFirstToSecond(TaskflowTaskOp first, TaskflowTaskOp second,
                                 bool permitJoins = false) {
  if (!first || !second || first == second ||
      first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) ||
      first->hasAttr("amoeba.control_predecessors") ||
      second->hasAttr("amoeba.control_predecessors"))
    return false;
  llvm::SmallPtrSet<Operation *, 8> intervening;
  for (Operation *operation = first->getNextNode();
       operation && operation != second.getOperation();
       operation = operation->getNextNode()) {
    if (isa<TaskflowChannelOp>(operation) ||
        (permitJoins && isa<TaskflowJoinOp>(operation)))
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

static void markTask(TaskflowTaskOp task, StringRef role, StringRef kind,
                     OpBuilder &builder) {
  task->setAttr(kSemanticRoleAttr, builder.getStringAttr(role));
  task->setAttr(kSemanticKindAttr, builder.getStringAttr(kind));
  task->setAttr(kRewriteSchemaAttr, builder.getStringAttr(kRewriteSchema));
}

static std::string appendAxis(StringRef axes, int64_t axis) {
  SmallVector<StringRef> fields;
  axes.split(fields, ',');
  std::string result = axes.str();
  std::string token = std::to_string(axis);
  if (llvm::is_contained(fields, StringRef(token)))
    return result;
  if (!result.empty())
    result += ",";
  result += token;
  return result;
}

static bool hasTiledAxis(TaskflowTaskOp task, int64_t axis) {
  auto attr = task->getAttrOfType<StringAttr>(kTiledAxesAttr);
  if (!attr)
    return false;
  SmallVector<StringRef> fields;
  attr.getValue().split(fields, ',');
  return llvm::is_contained(fields, StringRef(std::to_string(axis)));
}

static SmallVector<int64_t> counterBounds(TaskflowTaskOp task, bool upper) {
  SmallVector<int64_t> result;
  for (TaskflowCounterOp counter : counterChain(task)) {
    auto value = constantIndex(upper ? counter.getUpperBound()
                                     : counter.getLowerBound());
    if (!value)
      return {};
    result.push_back(*value);
  }
  return result;
}

// Attach the exact half-open region written by a static task.  The task-edge
// contract intentionally does not infer this fact from counter bounds alone:
// a task may use counters for an unrelated loop, or write a region that is
// wider than its iteration domain.  This pass has already proved the
// canonical state shape, so the counter bounds are the region contract for
// every output memref in this narrow rewrite family.  Keeping one descriptor
// per will_writes operand also makes the attribute survive task fusion where
// operand and result numbering changes.
static LogicalResult markOutputRegionContract(TaskflowTaskOp task,
                                              OpBuilder &builder) {
  if (!task || task.getWillWrites().empty() ||
      task.getOriginalWriteMemrefs().size() != task.getWillWrites().size())
    return failure();
  SmallVector<int64_t> lower = counterBounds(task, /*upper=*/false);
  SmallVector<int64_t> upper = counterBounds(task, /*upper=*/true);
  if (lower.empty() || upper.size() != lower.size())
    return failure();

  SmallVector<Attribute> lowers;
  SmallVector<Attribute> uppers;
  lowers.reserve(task.getWillWrites().size());
  uppers.reserve(task.getWillWrites().size());
  for (Value write : task.getWillWrites()) {
    auto type = dyn_cast<ShapedType>(write.getType());
    if (!type || !type.hasStaticShape() ||
        type.getRank() != static_cast<int64_t>(lower.size()))
      return failure();
    for (auto [dimension, bound] : llvm::enumerate(upper))
      if (bound > type.getShape()[dimension] || lower[dimension] < 0 ||
          bound <= lower[dimension])
        return failure();
    lowers.push_back(builder.getDenseI64ArrayAttr(lower));
    uppers.push_back(builder.getDenseI64ArrayAttr(upper));
  }
  task->setAttr(kTilingOutputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kTilingOutputRegionUppersAttr, builder.getArrayAttr(uppers));
  return success();
}

struct StaticAccessRegion {
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

static std::optional<StaticAccessRegion>
accessRegionForIndex(Value index, TaskflowTaskOp task,
                     TaskflowHyperblockOp hyperblock,
                     ArrayRef<TaskflowCounterOp> counters) {
  if (!hyperblock || !hyperblock.getBody().hasOneBlock())
    return std::nullopt;
  Block &hyperblockBody = hyperblock.getBody().front();
  for (auto [axis, argument] : llvm::enumerate(hyperblockBody.getArguments())) {
    if (index != argument)
      continue;
    if (axis >= counters.size())
      return std::nullopt;
    TaskflowCounterOp counter = counters[axis];
    auto lower = constantIndex(counter.getLowerBound());
    auto upper = constantIndex(counter.getUpperBound());
    if (!lower || !upper || *upper <= *lower)
      return std::nullopt;
    return StaticAccessRegion{{*lower}, {*upper}};
  }

  std::optional<StaticAccessRegion> result;
  task.walk([&](scf::ForOp loop) {
    if (result || index != loop.getInductionVar())
      return;
    auto lower = constantIndex(loop.getLowerBound());
    auto upper = constantIndex(loop.getUpperBound());
    if (!lower || !upper || *upper <= *lower)
      return;
    result = StaticAccessRegion{{*lower}, {*upper}};
  });
  return result;
}

// Infer rectangular read regions from the canonical load indices.  The
// producer's A/B/C reads have different projections of the M/N/K domains, so
// blindly clamping every operand to the output tile would be unsound.  This
// small index proof handles direct hyperblock counter arguments and the
// canonical sequential K loop; anything outside that proof remains an
// explicit tensor-wide fallback.
static SmallVector<std::optional<StaticAccessRegion>>
inferInputRegions(TaskflowTaskOp task) {
  SmallVector<std::optional<StaticAccessRegion>> result(
      task.getWillReads().size());
  if (!task || !task.getBody().hasOneBlock() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return result;
  TaskflowHyperblockOp hyperblock = findHyperblock(task);
  SmallVector<TaskflowCounterOp> counters = counterChain(task);
  if (!hyperblock || counters.empty())
    return result;
  Block &body = task.getBody().front();
  SmallVector<std::optional<StaticAccessRegion>> accesses;
  accesses.resize(task.getWillReads().size());
  SmallVector<bool> sawLoad(task.getWillReads().size(), false);
  bool invalid = false;
  task.walk([&](memref::LoadOp load) {
    if (invalid)
      return;
    auto memref = dyn_cast<BlockArgument>(load.getMemRef());
    if (!memref || memref.getOwner() != &body)
      return;
    unsigned readIndex = memref.getArgNumber();
    if (readIndex >= task.getWillReads().size())
      return;
    sawLoad[readIndex] = true;
    StaticAccessRegion combined;
    for (Value index : load.getIndices()) {
      std::optional<StaticAccessRegion> dimension =
          accessRegionForIndex(index, task, hyperblock, counters);
      if (!dimension || dimension->lower.size() != 1 ||
          dimension->upper.size() != 1) {
        invalid = true;
        return;
      }
      combined.lower.push_back(dimension->lower.front());
      combined.upper.push_back(dimension->upper.front());
    }
    if (!accesses[readIndex]) {
      accesses[readIndex] = std::move(combined);
      return;
    }
    StaticAccessRegion &previous = *accesses[readIndex];
    if (previous.lower.size() != combined.lower.size()) {
      invalid = true;
      return;
    }
    for (size_t dimension = 0; dimension < previous.lower.size(); ++dimension) {
      previous.lower[dimension] =
          std::min(previous.lower[dimension], combined.lower[dimension]);
      previous.upper[dimension] =
          std::max(previous.upper[dimension], combined.upper[dimension]);
    }
  });
  if (invalid)
    return SmallVector<std::optional<StaticAccessRegion>>(
        task.getWillReads().size());
  for (unsigned index = 0; index < result.size(); ++index) {
    if (!sawLoad[index] || !accesses[index])
      continue;
    auto type = dyn_cast<ShapedType>(task.getWillReads()[index].getType());
    if (!type || !type.hasStaticShape() ||
        accesses[index]->lower.size() != static_cast<size_t>(type.getRank())) {
      accesses[index].reset();
      continue;
    }
    bool valid = true;
    for (auto [dimension, upper] : llvm::enumerate(accesses[index]->upper))
      valid &= accesses[index]->lower[dimension] >= 0 &&
               upper > accesses[index]->lower[dimension] &&
               upper <= type.getShape()[dimension];
    if (valid)
      result[index] = std::move(accesses[index]);
  }
  return result;
}

static LogicalResult markInputRegionContract(TaskflowTaskOp task,
                                             OpBuilder &builder) {
  if (!task || task.getWillReads().empty() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return success();
  SmallVector<std::optional<StaticAccessRegion>> regions =
      inferInputRegions(task);
  SmallVector<Attribute> lowers;
  SmallVector<Attribute> uppers;
  SmallVector<Attribute> reasons;
  lowers.reserve(regions.size());
  uppers.reserve(regions.size());
  reasons.reserve(regions.size());
  for (const std::optional<StaticAccessRegion> &region : regions) {
    if (region) {
      lowers.push_back(builder.getDenseI64ArrayAttr(region->lower));
      uppers.push_back(builder.getDenseI64ArrayAttr(region->upper));
      reasons.push_back(builder.getStringAttr("proven_rectangular"));
    } else {
      lowers.push_back(builder.getUnitAttr());
      uppers.push_back(builder.getUnitAttr());
      reasons.push_back(builder.getStringAttr("unknown"));
    }
  }
  task->setAttr(kTilingInputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kTilingInputRegionUppersAttr, builder.getArrayAttr(uppers));
  task->setAttr(kTilingInputRegionReasonsAttr, builder.getArrayAttr(reasons));
  return success();
}

static SmallVector<TaskflowTaskOp> semanticTasks(func::FuncOp function,
                                                 StringRef role);

static void collectStateProducers(Value value,
                                  SmallVectorImpl<TaskflowTaskOp> &producers,
                                  DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    collectStateProducers(channel.getSource(), producers, visited);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates())
      collectStateProducers(state, producers, visited);
    return;
  }
  auto task = value.getDefiningOp<TaskflowTaskOp>();
  if (!task)
    return;
  if (llvm::is_contained(task.getDoneWrites(), value))
    producers.push_back(task);
}

static void readSemanticEdgeFields(TaskflowTaskOp task, StringRef &role,
                                   StringRef &scope) {
  auto annotations = task->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
  if (!annotations || annotations.empty())
    return;
  auto encoded = dyn_cast<StringAttr>(annotations[0]);
  if (!encoded)
    return;
  SmallVector<StringRef> fields;
  encoded.getValue().split(fields, '|');
  if (fields.size() == 3 && !fields[1].empty() && !fields[2].empty()) {
    role = fields[1];
    scope = fields[2];
  }
}

static std::optional<StaticAccessRegion>
readInputRegionContract(TaskflowTaskOp task, unsigned index) {
  auto lowers = task->getAttrOfType<ArrayAttr>(kTilingInputRegionLowersAttr);
  auto uppers = task->getAttrOfType<ArrayAttr>(kTilingInputRegionUppersAttr);
  if (!lowers || !uppers || index >= lowers.size() || index >= uppers.size())
    return std::nullopt;
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[index]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[index]);
  if (!lower || !upper)
    return std::nullopt;
  return StaticAccessRegion{SmallVector<int64_t>(lower.asArrayRef()),
                            SmallVector<int64_t>(upper.asArrayRef())};
}

static std::optional<StaticAccessRegion>
readOutputRegionContract(TaskflowTaskOp task, unsigned index) {
  auto lowers = task->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
  auto uppers = task->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
  if (!lowers || !uppers || index >= lowers.size() || index >= uppers.size())
    return std::nullopt;
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[index]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[index]);
  if (!lower || !upper)
    return std::nullopt;
  return StaticAccessRegion{SmallVector<int64_t>(lower.asArrayRef()),
                            SmallVector<int64_t>(upper.asArrayRef())};
}

static std::optional<StaticAccessRegion>
outputRegionForTask(TaskflowTaskOp task) {
  if (std::optional<StaticAccessRegion> region =
          readOutputRegionContract(task, 0))
    return region;
  SmallVector<int64_t> lower = counterBounds(task, /*upper=*/false);
  SmallVector<int64_t> upper = counterBounds(task, /*upper=*/true);
  if (lower.empty() || upper.size() != lower.size())
    return std::nullopt;
  return StaticAccessRegion{std::move(lower), std::move(upper)};
}

static bool regionsOverlap(const StaticAccessRegion &lhs,
                           const StaticAccessRegion &rhs) {
  if (lhs.lower.size() != lhs.upper.size() ||
      rhs.lower.size() != rhs.upper.size() ||
      lhs.lower.size() != rhs.lower.size())
    return true;
  for (size_t dimension = 0; dimension < lhs.lower.size(); ++dimension)
    if (lhs.upper[dimension] <= rhs.lower[dimension] ||
        rhs.upper[dimension] <= lhs.lower[dimension])
      return false;
  return true;
}

// M/N tiling can rename any state producer, including K partials and the
// final reduction-consumer fusion.  A consumer tile initially keeps a join as
// its input, so the SSA edge expands to the join's tiled predecessors.  Walk
// those predecessors directly and rewrite every existing semantic annotation
// to the names which still exist after cloning.  This is intentionally
// broader than producer/consumer roles: K reduction tasks use roles such as
// "reduction" and "fused_reduction_consumer" but obey the same edge contract.
static LogicalResult updateSemanticIncomingEdges(func::FuncOp function,
                                                 OpBuilder &builder) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) {
    auto role = task->getAttrOfType<StringAttr>(kSemanticRoleAttr);
    auto incoming = task->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
    bool hasIncoming = incoming && !incoming.empty();
    if (hasIncoming || (role && role.getValue() == "consumer"))
      tasks.push_back(task);
  });
  for (TaskflowTaskOp consumer : tasks) {
    StringRef role = "producer_consumer";
    StringRef scope = "tile_local";
    readSemanticEdgeFields(consumer, role, scope);
    SmallVector<std::optional<StaticAccessRegion>> inferredInputs =
        inferInputRegions(consumer);
    SmallVector<Attribute> annotations;
    for (auto [inputIndex, input] : llvm::enumerate(consumer.getWillReads())) {
      if (inputIndex >= consumer.getOriginalReadMemrefs().size())
        continue;
      SmallVector<TaskflowTaskOp> candidates;
      DenseSet<Value> visited;
      collectStateProducers(input, candidates, visited);
      Value consumerRoot = consumer.getOriginalReadMemrefs()[inputIndex];
      for (TaskflowTaskOp producer : candidates) {
        if (!producer || producer.getOriginalWriteMemrefs().size() != 1 ||
            producer.getOriginalWriteMemrefs().front() != consumerRoot ||
            producer.getTaskName() == consumer.getTaskName())
          continue;
        std::optional<StaticAccessRegion> consumerRegion =
            readInputRegionContract(consumer, inputIndex);
        if (!consumerRegion && inputIndex < inferredInputs.size())
          consumerRegion = inferredInputs[inputIndex];
        std::optional<StaticAccessRegion> producerRegion =
            outputRegionForTask(producer);
        // Unknown regions remain conservative: the facts builder will retain
        // the corresponding tensor-wide edge.  When both endpoint regions
        // are known, only an intersecting producer is an actual edge of a
        // joined tile state.
        if (consumerRegion && producerRegion &&
            !regionsOverlap(*consumerRegion, *producerRegion))
          continue;
        annotations.push_back(builder.getStringAttr(
            (Twine(producer.getTaskName()) + "|" + role + "|" + scope).str()));
      }
    }
    if (!annotations.empty()) {
      consumer->setAttr(kSemanticIncomingEdgesAttr,
                        builder.getArrayAttr(annotations));
    } else if (consumer->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr) &&
               !consumer->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr)
                    .empty()) {
      // Preserve fail-closed behavior if an annotated canonical edge cannot
      // be mapped to a derived tile result.
      return reject(consumer,
                    "cannot remap semantic incoming edge to a tiled producer");
    }
  }
  return success();
}

static void collectStateProducers(Value value,
                                  SmallVectorImpl<TaskflowTaskOp> &producers,
                                  DenseSet<Value> &visited);

static bool provesSequentialKStateCarry(TaskflowTaskOp task) {
  if (task.getWillReads().size() != 3 ||
      task.getOriginalReadMemrefs().size() != 3 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      task.getOriginalReadMemrefs()[2] != task.getOriginalWriteMemrefs()[0] ||
      !task->hasAttr("amoeba.semantic.k_tiled"))
    return false;
  auto policy = task->getAttrOfType<StringAttr>("amoeba.semantic.k_policy");
  auto index =
      task->getAttrOfType<IntegerAttr>("amoeba.semantic.k_block_index");
  auto count =
      task->getAttrOfType<IntegerAttr>("amoeba.semantic.k_block_count");
  auto kRange = task->getAttrOfType<DenseI64ArrayAttr>(kSemanticKRangeAttr);
  if (!policy || policy.getValue() != "sequential" || !index || !count ||
      index.getInt() <= 0 || index.getInt() >= count.getInt() || !kRange ||
      kRange.asArrayRef().size() != 2 ||
      kRange.asArrayRef()[0] >= kRange.asArrayRef()[1])
    return false;
  SmallVector<TaskflowTaskOp> predecessors;
  DenseSet<Value> visited;
  collectStateProducers(task.getWillReads()[2], predecessors, visited);
  if (predecessors.empty())
    return false;
  for (TaskflowTaskOp predecessor : predecessors) {
    auto previousIndex = predecessor->getAttrOfType<IntegerAttr>(
        "amoeba.semantic.k_block_index");
    auto previousCount = predecessor->getAttrOfType<IntegerAttr>(
        "amoeba.semantic.k_block_count");
    auto previousRange =
        predecessor->getAttrOfType<DenseI64ArrayAttr>(kSemanticKRangeAttr);
    if (!previousIndex || previousIndex.getInt() + 1 != index.getInt() ||
        !previousCount || previousCount.getInt() != count.getInt() ||
        !previousRange || previousRange.asArrayRef().size() != 2 ||
        previousRange.asArrayRef()[1] != kRange.asArrayRef()[0] ||
        predecessor.getOriginalWriteMemrefs().size() != 1 ||
        predecessor.getOriginalWriteMemrefs()[0] !=
            task.getOriginalWriteMemrefs()[0])
      return false;
  }
  return true;
}

static bool taskHasCanonicalStateShape(TaskflowTaskOp task) {
  if (!task || !task.getBody().hasOneBlock() ||
      task.getDoneWrites().size() != 1 || task.getWillWrites().size() != 1 ||
      ((!task.getDoneReads().empty() &&
        task.getDoneReads().size() != task.getWillReads().size())) ||
      task.getValueOutputs().size() != 0 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return false;
  auto counters = counterChain(task);
  auto hyperblock = findHyperblock(task);
  if (counters.empty() || !hyperblock || !hyperblock.getIterArgs().empty() ||
      hyperblock.getNumResults() != 0 ||
      hyperblock.getIndices().size() != counters.size())
    return false;
  for (auto [index, counter] : llvm::enumerate(counters)) {
    auto lower = constantIndex(counter.getLowerBound());
    auto upper = constantIndex(counter.getUpperBound());
    auto step = constantIndex(counter.getStep());
    bool parentMismatch =
        index == 0
            ? static_cast<bool>(counter.getParentIndex())
            : counter.getParentIndex() != counters[index - 1].getCounterIndex();
    if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
        *step != 1 || parentMismatch)
      return false;
    if (hyperblock.getIndices()[index] != counter.getCounterIndex())
      return false;
  }
  SmallVector<Value> states(task.getWillReads());
  states.push_back(task.getWillWrites().front());
  for (Value state : states) {
    auto type = dyn_cast<MemRefType>(state.getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        type.getRank() != static_cast<int64_t>(counters.size()))
      return false;
  }
  SmallVector<Value> roots(task.getOriginalReadMemrefs());
  roots.push_back(task.getOriginalWriteMemrefs().front());
  for (auto [state, root] : llvm::zip(states, roots))
    if (state.getType() != root.getType())
      return false;
  bool provenStateCarry = provesSequentialKStateCarry(task);
  for (size_t first = 0; first < roots.size(); ++first)
    for (size_t second = first + 1; second < roots.size(); ++second)
      if (!provesDistinctStorage(roots[first], roots[second]) &&
          !(provenStateCarry && first == 2 && second + 1 == roots.size() &&
            roots[first] == roots[second]))
        return false;
  return true;
}

struct CanonicalPair {
  func::FuncOp function;
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
};

static FailureOr<CanonicalPair> findCanonicalPair(ModuleOp module,
                                                  StringRef functionName,
                                                  bool requirePair = true) {
  SmallVector<func::FuncOp> functions;
  module.walk([&](func::FuncOp function) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask &&
        (functionName.empty() || function.getSymName() == functionName))
      functions.push_back(function);
  });
  if (functions.size() != 1) {
    module.emitError()
        << "joint semantic rewrite requires exactly one selected "
           "Taskflow function (found "
        << functions.size() << ")";
    return failure();
  }
  CanonicalPair pair{functions.front(), nullptr, nullptr};
  SmallVector<TaskflowTaskOp> tasks;
  pair.function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  if (requirePair) {
    if (tasks.size() != 2) {
      pair.function.emitError()
          << "joint semantic rewrite expects exactly two Taskflow tasks; "
             "use a per-task tiling pass after each graph transition";
      return failure();
    }
    for (TaskflowTaskOp task : tasks) {
      if (task.getTaskName() == "producer")
        pair.producer = task;
      else if (task.getTaskName() == "consumer")
        pair.consumer = task;
    }
    if (!pair.producer || !pair.consumer) {
      pair.function.emitError()
          << "canonical pair must contain @producer and @consumer tasks";
      return failure();
    }
    if (!taskHasCanonicalStateShape(pair.producer) ||
        !taskHasCanonicalStateShape(pair.consumer)) {
      pair.function.emitError()
          << "canonical pair has unsupported Taskflow state shape";
      return failure();
    }
    if (!sameLoopStructure(pair.producer, pair.consumer)) {
      pair.function.emitError()
          << "producer and consumer loop domains are not identical";
      return failure();
    }
    unsigned relationCount = 0;
    for (Value input : pair.consumer.getWillReads()) {
      Value source = peelChannels(input);
      if (llvm::is_contained(pair.producer.getDoneWrites(), source))
        ++relationCount;
    }
    if (relationCount != 1) {
      pair.function.emitError()
          << "canonical pair needs exactly one producer-consumer RAW state";
      return failure();
    }
  }
  return pair;
}

//===----------------------------------------------------------------------===//
// M/N task tiling
//===----------------------------------------------------------------------===//

struct TilePlan {
  TaskflowTaskOp source;
  int64_t axis = 0;
  int64_t tileSize = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  SmallVector<TaskflowCounterOp> counters;
};

static FailureOr<TilePlan> analyzeTile(TaskflowTaskOp task, int64_t axis,
                                       int64_t tileSize) {
  auto role = task->getAttrOfType<StringAttr>(kSemanticRoleAttr);
  if (role &&
      (role.getValue() == "reduction" ||
       role.getValue() == "fused_reduction_consumer") &&
      !task->hasAttr("amoeba.semantic.numeric_reduction"))
    return reject(task,
                  "M/N tiling of a reduction task requires the proven numeric "
                  "reduction contract");
  if (!taskHasCanonicalStateShape(task))
    return reject(task,
                  "semantic tiling requires a canonical static Taskflow task");
  TilePlan plan{task, axis, tileSize, 0, 0, counterChain(task)};
  if (axis < 0 || axis >= static_cast<int64_t>(plan.counters.size()) ||
      tileSize < 2)
    return reject(task, "tile axis must select M/N and tile-size must be >= 2");
  auto lower = constantIndex(plan.counters[axis].getLowerBound());
  auto upper = constantIndex(plan.counters[axis].getUpperBound());
  if (!lower || !upper || *upper <= *lower || tileSize >= *upper - *lower)
    return reject(task, "tile-size must split a static non-empty domain");
  plan.lower = *lower;
  plan.upper = *upper;
  if (hasTiledAxis(task, axis))
    return reject(task, "the selected M/N axis is already tiled");
  return plan;
}

static TaskflowTaskOp cloneTile(OpBuilder &builder, const TilePlan &plan,
                                int64_t part, int64_t lower, int64_t upper) {
  TaskflowTaskOp source = plan.source;
  std::string name = (Twine(source.getTaskName()) + ".tile." +
                      Twine(plan.axis) + "." + Twine(part))
                         .str();
  auto tile = builder.create<TaskflowTaskOp>(
      source.getLoc(), source.getDoneReads().getTypes(),
      source.getDoneWrites().getTypes(), TypeRange{}, source.getWillReads(),
      source.getWillWrites(), source.getValueInputs(),
      builder.getStringAttr(name), source.getOriginalReadMemrefs(),
      source.getOriginalWriteMemrefs());
  Block *body = new Block();
  tile.getBody().push_back(body);
  IRMapping mapping;
  Block &sourceBody = source.getBody().front();
  for (auto [index, operand] : llvm::enumerate(source->getOperands())) {
    if (index >= sourceBody.getNumArguments())
      break;
    mapping.map(sourceBody.getArgument(index),
                body->addArgument(operand.getType(), source.getLoc()));
  }
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : sourceBody)
    if (!isa<TaskflowYieldOp>(&operation))
      bodyBuilder.clone(operation, mapping);
  SmallVector<TaskflowCounterOp> clonedCounters;
  for (Operation &operation : *body)
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      clonedCounters.push_back(counter);
  if (clonedCounters.size() != plan.counters.size())
    return TaskflowTaskOp();
  bodyBuilder.setInsertionPoint(clonedCounters[plan.axis]);
  Value lowerValue =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
  Value upperValue =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
  clonedCounters[plan.axis].getLowerBoundMutable().assign(lowerValue);
  clonedCounters[plan.axis].getUpperBoundMutable().assign(upperValue);
  bodyBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> readResults;
  if (!source.getDoneReads().empty())
    for (unsigned index = 0; index < source.getWillReads().size(); ++index)
      readResults.push_back(body->getArgument(index));
  bodyBuilder.create<TaskflowYieldOp>(
      source.getLoc(), readResults,
      ValueRange{body->getArgument(source.getWillReads().size())},
      ValueRange{});
  markTask(tile,
           source->getAttrOfType<StringAttr>(kSemanticRoleAttr)
               ? source->getAttrOfType<StringAttr>(kSemanticRoleAttr).getValue()
               : (source.getTaskName().starts_with("consumer") ? "consumer"
                                                               : "producer"),
           source->getAttrOfType<StringAttr>(kSemanticKindAttr)
               ? source->getAttrOfType<StringAttr>(kSemanticKindAttr).getValue()
               : (source.getTaskName().starts_with("consumer")
                      ? "elementwise_consumer"
                      : "gemm_producer"),
           builder);
  // Preserve semantic proof and K-policy attributes when the source is a
  // numeric reduction or a fused reduction-consumer.  The tile-specific
  // descriptors below replace the old ranges; all other semantic facts,
  // including numeric_reduction/reduction_final and incoming_edges, remain
  // valid and are remapped after the complete axis pass.
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (!name.starts_with("amoeba.semantic.") || name == kTiledAxesAttr ||
        name == kTileAxisAttr || name == kTileSizeAttr ||
        name == kTileRangeAttr)
      continue;
    tile->setAttr(attr.getName(), attr.getValue());
  }
  tile->setAttr(
      kTiledAxesAttr,
      builder.getStringAttr(appendAxis(
          source->getAttrOfType<StringAttr>(kTiledAxesAttr)
              ? source->getAttrOfType<StringAttr>(kTiledAxesAttr).getValue()
              : StringRef(),
          plan.axis)));
  tile->setAttr(kTileAxisAttr, builder.getI64IntegerAttr(plan.axis));
  tile->setAttr(kTileSizeAttr, builder.getI64IntegerAttr(plan.tileSize));
  tile->setAttr(kTileRangeAttr, builder.getDenseI64ArrayAttr({lower, upper}));
  tile->setAttr("amoeba.tiling.parent_task",
                builder.getStringAttr(source.getTaskName()));
  tile->setAttr("amoeba.tiling.axis", builder.getI64IntegerAttr(plan.axis));
  tile->setAttr(
      "amoeba.tiling.factor",
      builder.getI64IntegerAttr((plan.upper - plan.lower + plan.tileSize - 1) /
                                plan.tileSize));
  tile->setAttr("amoeba.tiling.part_index", builder.getI64IntegerAttr(part));
  // The bounds above are now the task's proven output region.  A later graph
  // facts pass must be able to distinguish disjoint tile writes without
  // falling back to source-order WAW edges.
  if (failed(markOutputRegionContract(tile, builder)))
    return TaskflowTaskOp();
  if (failed(markInputRegionContract(tile, builder)))
    return TaskflowTaskOp();
  return tile;
}

static LogicalResult tileTask(TilePlan plan, func::FuncOp function) {
  int64_t count = (plan.upper - plan.lower + plan.tileSize - 1) / plan.tileSize;
  SmallVector<std::string> names;
  for (int64_t part = 0; part < count; ++part)
    names.push_back((Twine(plan.source.getTaskName()) + ".tile." +
                     Twine(plan.axis) + "." + Twine(part))
                        .str());
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    if (task == plan.source)
      return;
    collision |= llvm::is_contained(names, task.getTaskName().str());
  });
  if (collision)
    return reject(plan.source, "tile-derived task name already exists");

  OpBuilder builder(plan.source);
  SmallVector<Value> tileStates;
  int64_t cursor = plan.lower;
  for (int64_t part = 0; part < count; ++part) {
    int64_t end = std::min(cursor + plan.tileSize, plan.upper);
    TaskflowTaskOp tile = cloneTile(builder, plan, part, cursor, end);
    if (!tile)
      return reject(plan.source, "failed to clone the canonical task body");
    tileStates.push_back(tile.getDoneWrites().front());
    cursor = end;
  }
  SmallVector<int64_t> regionLower = counterBounds(plan.source, false);
  SmallVector<int64_t> regionUpper = counterBounds(plan.source, true);
  if (regionLower.empty() || regionUpper.empty())
    return reject(plan.source, "cannot derive a static join region");
  auto join = builder.create<TaskflowJoinOp>(
      plan.source.getLoc(), plan.source.getDoneWrites().front().getType(),
      tileStates, plan.source.getWillWrites().front(),
      builder.getI64IntegerAttr(plan.axis),
      builder.getDenseI64ArrayAttr(regionLower),
      builder.getDenseI64ArrayAttr(regionUpper));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  plan.source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  plan.source.erase();
  return success();
}

static SmallVector<TaskflowTaskOp> semanticTasks(func::FuncOp function,
                                                 StringRef role) {
  SmallVector<TaskflowTaskOp> result;
  function.walk([&](TaskflowTaskOp task) {
    auto roleAttr = task->getAttrOfType<StringAttr>(kSemanticRoleAttr);
    if (roleAttr && roleAttr.getValue() == role)
      result.push_back(task);
  });
  return result;
}

static void inferAndMarkCanonicalRoles(func::FuncOp function,
                                       OpBuilder &builder) {
  function.walk([&](TaskflowTaskOp task) {
    auto roleAttr = task->getAttrOfType<StringAttr>(kSemanticRoleAttr);
    auto kindAttr = task->getAttrOfType<StringAttr>(kSemanticKindAttr);
    StringRef role =
        roleAttr ? roleAttr.getValue()
                 : (task.getTaskName().starts_with("consumer") ? "consumer"
                                                               : "producer");
    StringRef kind = kindAttr ? kindAttr.getValue()
                              : (role == "consumer" ? "elementwise_consumer"
                                                    : "gemm_producer");
    markTask(task, role, kind, builder);
  });
}

static LogicalResult tileAxis(ModuleOp module, StringRef functionName,
                              int64_t axis, int64_t tileSize) {
  FailureOr<CanonicalPair> selected =
      findCanonicalPair(module, functionName, /*requirePair=*/false);
  if (failed(selected))
    return failure();
  func::FuncOp function = selected->function;
  OpBuilder marker(function.getContext());
  inferAndMarkCanonicalRoles(function, marker);
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) {
    auto role = task->getAttrOfType<StringAttr>(kSemanticRoleAttr);
    if (role &&
        (role.getValue() == "producer" || role.getValue() == "consumer" ||
         role.getValue() == "fused_producer_consumer" ||
         (role.getValue() == "reduction" &&
          task->hasAttr("amoeba.semantic.numeric_reduction")) ||
         (role.getValue() == "fused_reduction_consumer" &&
          task->hasAttr("amoeba.semantic.numeric_reduction"))))
      tasks.push_back(task);
  });
  if (tasks.empty())
    return reject(function, "no semantic producer/consumer tasks to tile");
  // Plans are analyzed before mutation.  This makes the pass atomic for the
  // common error cases and prevents one task from being tiled before another
  // task is discovered to be illegal.
  SmallVector<TilePlan> plans;
  for (TaskflowTaskOp task : tasks) {
    FailureOr<TilePlan> plan = analyzeTile(task, axis, tileSize);
    if (failed(plan))
      return failure();
    plans.push_back(*plan);
  }
  // Process in reverse block order so insertion before a later task cannot
  // invalidate handles of an earlier plan.
  for (auto it = plans.rbegin(); it != plans.rend(); ++it)
    if (failed(tileTask(*it, function)))
      return failure();
  if (failed(updateSemanticIncomingEdges(function, marker)))
    return failure();
  function->setAttr(kRewriteSchemaAttr, marker.getStringAttr(kRewriteSchema));
  return success();
}

//===----------------------------------------------------------------------===//
// Producer/consumer fusion
//===----------------------------------------------------------------------===//

static Value resolveThrough(Value value, TaskflowTaskOp task) {
  for (auto [index, result] : llvm::enumerate(task.getDoneReads()))
    if (value == result)
      return task.getWillReads()[index];
  for (auto [index, result] : llvm::enumerate(task.getDoneWrites()))
    if (value == result)
      return task.getWillWrites()[index];
  return value;
}

static bool hasInterveningUses(TaskflowTaskOp producer,
                               TaskflowTaskOp consumer) {
  llvm::SmallPtrSet<Value, 8> results;
  for (Value value : producer.getDoneReads())
    results.insert(value);
  for (Value value : producer.getDoneWrites())
    results.insert(value);
  for (Value value : producer.getValueOutputs())
    results.insert(value);
  bool inRange = false;
  for (Operation &operation : *producer->getBlock()) {
    if (&operation == producer.getOperation()) {
      inRange = true;
      continue;
    }
    if (&operation == consumer.getOperation())
      break;
    if (inRange && !isa<TaskflowChannelOp, TaskflowJoinOp>(&operation))
      for (Value operand : operation.getOperands())
        if (results.contains(operand))
          return true;
  }
  return false;
}

static void cloneBodyMisc(Block &source, OpBuilder &builder,
                          IRMapping &mapping) {
  for (Operation &operation : source)
    if (!isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(
            &operation))
      builder.clone(operation, mapping);
}

static scf::ForOp directInnerFor(TaskflowHyperblockOp hyperblock) {
  scf::ForOp result;
  if (!hyperblock)
    return result;
  for (Operation &operation : hyperblock.getBody().front()) {
    auto loop = dyn_cast<scf::ForOp>(&operation);
    if (!loop)
      continue;
    if (result)
      return scf::ForOp();
    result = loop;
  }
  return result;
}

static bool scfBoundsMatch(scf::ForOp lhs, scf::ForOp rhs) {
  if (!lhs || !rhs)
    return false;
  auto lhsLower = constantIndex(lhs.getLowerBound());
  auto rhsLower = constantIndex(rhs.getLowerBound());
  auto lhsUpper = constantIndex(lhs.getUpperBound());
  auto rhsUpper = constantIndex(rhs.getUpperBound());
  auto lhsStep = constantIndex(lhs.getStep());
  auto rhsStep = constantIndex(rhs.getStep());
  return lhsLower && rhsLower && lhsUpper && rhsUpper && lhsStep && rhsStep &&
         *lhsLower == *rhsLower && *lhsUpper == *rhsUpper &&
         *lhsStep == *rhsStep;
}

static LogicalResult replaceFusionResults(TaskflowTaskOp producer,
                                          TaskflowTaskOp consumer,
                                          Value intermediate,
                                          TaskflowTaskOp fused) {
  // Fused read states are newly exposed results.  Only old read states that
  // were explicitly yielded by either task need replacement; the producer's
  // RAW write state is the eliminated intermediate and is intentionally not
  // paired with a fused result.
  auto replaceReadStates = [&](TaskflowTaskOp task, bool resolve) {
    for (unsigned index = 0; index < task.getDoneReads().size(); ++index) {
      Value original = task.getWillReads()[index];
      if (original == intermediate)
        continue;
      if (resolve)
        original = resolveThrough(original, producer);
      for (unsigned fusedIndex = 0; fusedIndex < fused.getWillReads().size();
           ++fusedIndex)
        if (fused.getWillReads()[fusedIndex] == original) {
          task.getDoneReads()[index].replaceAllUsesWith(
              fused.getDoneReads()[fusedIndex]);
          break;
        }
    }
  };
  replaceReadStates(producer, /*resolve=*/false);
  replaceReadStates(consumer, /*resolve=*/true);
  if (consumer.getDoneWrites().size() != fused.getDoneWrites().size() ||
      consumer.getValueOutputs().size() != fused.getValueOutputs().size())
    return reject(consumer,
                  "consumer result arity changed during legalization");
  for (auto [oldValue, newValue] :
       llvm::zip(consumer.getDoneWrites(), fused.getDoneWrites()))
    oldValue.replaceAllUsesWith(newValue);
  for (auto [oldValue, newValue] :
       llvm::zip(consumer.getValueOutputs(), fused.getValueOutputs()))
    oldValue.replaceAllUsesWith(newValue);
  return success();
}

// This is a deliberately small structural fusion builder.  It is based on
// SSA values and not on textual names, so a channel can be present between
// the producer and consumer.  The only erased operation that is special is
// the producer store into the single-use intermediate; the consumer load from
// that same state is replaced by the stored SSA value.
static TaskflowTaskOp buildProducerConsumerFusion(TaskflowTaskOp producer,
                                                  TaskflowTaskOp consumer,
                                                  Value producerResult,
                                                  Value consumerInput,
                                                  OpBuilder &builder) {
  Location loc = consumer.getLoc();
  SmallVector<Value> producerReads(producer.getWillReads());
  SmallVector<Value> producerWrites(producer.getWillWrites());
  SmallVector<Value> producerValues(producer.getValueInputs());
  SmallVector<Value> consumerReads(consumer.getWillReads());
  SmallVector<Value> consumerWrites(consumer.getWillWrites());
  SmallVector<Value> consumerValues(consumer.getValueInputs());
  if (producerWrites.size() != 1 || consumerWrites.size() != 1 ||
      producerResult != producer.getDoneWrites().front() ||
      consumerInput != consumer.getWillReads().front())
    return TaskflowTaskOp();

  SmallVector<Value> fusedReads(producerReads);
  for (Value value : consumerReads)
    if (value != consumerInput)
      fusedReads.push_back(resolveThrough(value, producer));
  SmallVector<Value> fusedWrites(consumerWrites);
  SmallVector<Value> fusedValues(producerValues);
  for (Value value : consumerValues)
    fusedValues.push_back(resolveThrough(value, producer));

  SmallVector<Type> readTypes;
  for (Value value : fusedReads)
    readTypes.push_back(value.getType());
  SmallVector<Value> originalReads;
  for (Value value : producer.getOriginalReadMemrefs())
    originalReads.push_back(value);
  for (Value value : consumer.getOriginalReadMemrefs())
    if (value != producer.getOriginalWriteMemrefs().front() &&
        !llvm::is_contained(originalReads, value))
      originalReads.push_back(value);
  SmallVector<Value> originalWrites(consumer.getOriginalWriteMemrefs());

  auto fused = builder.create<TaskflowTaskOp>(
      loc, readTypes, consumer.getDoneWrites().getTypes(),
      consumer.getValueOutputs().getTypes(), fusedReads, fusedWrites,
      fusedValues,
      builder.getStringAttr((Twine(producer.getTaskName()) + ".fuse." +
                             Twine(consumer.getTaskName()))
                                .str()),
      originalReads, originalWrites);
  for (StringRef name :
       {"amoeba.semantic.K_range", "amoeba.semantic.k_range",
        "amoeba.semantic.k_policy", "amoeba.semantic.reduction_topology"})
    if (Attribute value = producer->getAttr(name))
      fused->setAttr(name, value);
  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value value : fusedReads)
    body->addArgument(value.getType(), loc);
  for (Value value : fusedWrites)
    body->addArgument(value.getType(), loc);
  for (Value value : fusedValues)
    body->addArgument(value.getType(), loc);

  IRMapping producerMapping;
  IRMapping consumerMapping;
  Block &producerBody = producer.getBody().front();
  Block &consumerBody = consumer.getBody().front();
  for (unsigned index = 0; index < producerReads.size(); ++index)
    producerMapping.map(producerBody.getArgument(index),
                        body->getArgument(index));
  // The sole producer write is an eliminated intermediate.  It deliberately
  // has no fused task operand.  Its store is removed below and its block
  // argument is therefore never looked up.
  for (unsigned index = 0; index < producerValues.size(); ++index)
    producerMapping.map(
        producerBody.getArgument(producerReads.size() + producerWrites.size() +
                                 index),
        body->getArgument(fusedReads.size() + fusedWrites.size() + index));

  unsigned consumerReadBase = producerReads.size();
  unsigned consumerValueBase =
      fusedReads.size() + fusedWrites.size() + producerValues.size();
  unsigned consumerWriteBase = fusedReads.size();
  unsigned nextConsumerRead = 0;
  for (unsigned index = 0; index < consumerReads.size(); ++index) {
    if (consumerReads[index] == consumerInput) {
      // The intermediate is represented by forwarded SSA below.
      continue;
    }
    consumerMapping.map(consumerBody.getArgument(index),
                        body->getArgument(consumerReadBase + nextConsumerRead));
    ++nextConsumerRead;
  }
  for (unsigned index = 0; index < consumerWrites.size(); ++index)
    consumerMapping.map(consumerBody.getArgument(consumerReads.size() + index),
                        body->getArgument(consumerWriteBase + index));
  for (unsigned index = 0; index < consumerValues.size(); ++index)
    consumerMapping.map(consumerBody.getArgument(consumerReads.size() +
                                                 consumerWrites.size() + index),
                        body->getArgument(consumerValueBase + index));

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  cloneBodyMisc(producerBody, builder, producerMapping);
  cloneBodyMisc(consumerBody, builder, consumerMapping);

  auto producerCounters = counterChain(producer);
  auto consumerCounters = counterChain(consumer);
  if (!producerCounters.empty() &&
      !sameCounterDomain(producerCounters, consumerCounters)) {
    fused.erase();
    return TaskflowTaskOp();
  }
  for (TaskflowCounterOp counter : producerCounters)
    builder.clone(*counter, producerMapping);
  auto producerHyperblock = findHyperblock(producer);
  auto consumerHyperblock = findHyperblock(consumer);
  if (!producerHyperblock || !consumerHyperblock) {
    fused.erase();
    return TaskflowTaskOp();
  }
  SmallVector<Value> triggers;
  for (Value index : producerHyperblock.getIndices())
    triggers.push_back(producerMapping.lookupOrDefault(index));
  auto fusedHyperblock = builder.create<TaskflowHyperblockOp>(
      loc, TypeRange{}, triggers, ValueRange{});
  Block *hyperblockBody = &fusedHyperblock.getBody().emplaceBlock();
  for (BlockArgument argument :
       producerHyperblock.getBody().front().getArguments())
    hyperblockBody->addArgument(argument.getType(), loc);
  for (auto [oldArgument, newArgument] :
       llvm::zip(producerHyperblock.getBody().front().getArguments(),
                 hyperblockBody->getArguments()))
    producerMapping.map(oldArgument, newArgument);
  for (auto [oldArgument, newArgument] :
       llvm::zip(consumerHyperblock.getBody().front().getArguments(),
                 hyperblockBody->getArguments()))
    consumerMapping.map(oldArgument, newArgument);

  Block &producerHyperbody = producerHyperblock.getBody().front();
  Block &consumerHyperbody = consumerHyperblock.getBody().front();
  scf::ForOp producerFor = directInnerFor(producerHyperblock);
  scf::ForOp consumerFor = directInnerFor(consumerHyperblock);
  Value forwarded;
  {
    OpBuilder::InsertionGuard hyperGuard(builder);
    builder.setInsertionPointToStart(hyperblockBody);
    if (producerFor && consumerFor) {
      if (!scfBoundsMatch(producerFor, consumerFor)) {
        fused.erase();
        return TaskflowTaskOp();
      }
      for (Operation &operation : producerHyperbody)
        if (!isa<TaskflowHyperblockYieldOp, scf::ForOp>(&operation))
          builder.clone(operation, producerMapping);
      for (Operation &operation : consumerHyperbody)
        if (!isa<TaskflowHyperblockYieldOp, scf::ForOp>(&operation))
          builder.clone(operation, consumerMapping);
      Value lower =
          producerMapping.lookupOrDefault(producerFor.getLowerBound());
      Value upper =
          producerMapping.lookupOrDefault(producerFor.getUpperBound());
      Value step = producerMapping.lookupOrDefault(producerFor.getStep());
      auto merged = builder.create<scf::ForOp>(loc, lower, upper, step);
      producerMapping.map(producerFor.getInductionVar(),
                          merged.getInductionVar());
      consumerMapping.map(consumerFor.getInductionVar(),
                          merged.getInductionVar());
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToEnd(merged.getBody());
      for (Operation &operation : *producerFor.getBody()) {
        if (isa<scf::YieldOp>(&operation))
          continue;
        if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
          auto intermediateArgument =
              producerBody.getArgument(producerReads.size());
          if (store.getMemRef() == intermediateArgument) {
            forwarded =
                producerMapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(operation, producerMapping);
      }
      if (!forwarded) {
        fused.erase();
        return TaskflowTaskOp();
      }
      for (Operation &operation : *consumerFor.getBody()) {
        if (isa<scf::YieldOp>(&operation))
          continue;
        if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
          auto intermediateArgument = consumerBody.getArgument(0);
          if (load.getMemRef() == intermediateArgument) {
            consumerMapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(operation, consumerMapping);
      }
    } else {
      for (Operation &operation : producerHyperbody) {
        if (isa<TaskflowHyperblockYieldOp>(&operation))
          continue;
        if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
          auto intermediateArgument =
              producerBody.getArgument(producerReads.size());
          if (store.getMemRef() == intermediateArgument) {
            forwarded =
                producerMapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(operation, producerMapping);
      }
      if (!forwarded) {
        fused.erase();
        return TaskflowTaskOp();
      }
      for (Operation &operation : consumerHyperbody) {
        if (isa<TaskflowHyperblockYieldOp>(&operation))
          continue;
        if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
          auto intermediateArgument = consumerBody.getArgument(0);
          if (load.getMemRef() == intermediateArgument) {
            consumerMapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(operation, consumerMapping);
      }
    }
    builder.create<TaskflowHyperblockYieldOp>(loc);
  }
  SmallVector<Value> readResults;
  for (auto [index, value] : llvm::enumerate(fusedReads))
    readResults.push_back(body->getArgument(index));
  SmallVector<Value> writeResults;
  auto consumerYield = cast<TaskflowYieldOp>(consumerBody.getTerminator());
  for (Value value : consumerYield.getDoneWrites())
    writeResults.push_back(consumerMapping.lookupOrDefault(value));
  SmallVector<Value> valueResults;
  for (Value value : consumerYield.getValueResults())
    valueResults.push_back(consumerMapping.lookupOrDefault(value));
  builder.setInsertionPointToEnd(body);
  builder.create<TaskflowYieldOp>(loc, readResults, writeResults, valueResults);
  markTask(fused, "fused_producer_consumer", "fused_producer_consumer",
           builder);
  fused->setAttr(kFusionModeAttr, builder.getStringAttr("producer_consumer"));
  // The consumer's static iteration domain is retained by the fused body and
  // is the exact output region for its D write.  Re-materialize the contract
  // on the new task because attributes are not copied by the generic builder.
  if (failed(markOutputRegionContract(fused, builder))) {
    fused.erase();
    return TaskflowTaskOp();
  }
  if (failed(markInputRegionContract(fused, builder))) {
    fused.erase();
    return TaskflowTaskOp();
  }
  (void)consumer;
  return fused;
}

static LogicalResult fusePair(TaskflowTaskOp producer, TaskflowTaskOp consumer,
                              bool permitJoins = false) {
  if (!canMoveFirstToSecond(producer, consumer, permitJoins) ||
      !sameLoopStructure(producer, consumer) ||
      hasInterveningUses(producer, consumer))
    return reject(consumer,
                  "producer-consumer fusion failed ordering or loop proof");
  SmallVector<TaskflowChannelOp> channels;
  Value producerResult;
  Value consumerInput;
  for (Value input : consumer.getWillReads()) {
    Value source = peelChannels(input);
    if (!llvm::is_contained(producer.getDoneWrites(), source))
      continue;
    if (!exclusiveChannelPath(source, input, channels))
      continue;
    producerResult = source;
    consumerInput = input;
    break;
  }
  if (!producerResult || !consumerInput)
    return reject(consumer, "fusion requires one exclusive producer RAW state");
  for (Value result : producer.getDoneWrites())
    if (result != producerResult && !result.use_empty())
      return reject(producer, "producer has another live write state");
  if (!producer.getValueOutputs().empty() ||
      !consumer.getValueOutputs().empty())
    return reject(consumer,
                  "semantic fusion currently requires no value outputs");
  OpBuilder builder(consumer);
  TaskflowTaskOp fused = buildProducerConsumerFusion(
      producer, consumer, producerResult, consumerInput, builder);
  if (!fused)
    return reject(consumer, "failed to clone producer and consumer bodies");
  if (failed(replaceFusionResults(producer, consumer, producerResult, fused))) {
    fused.erase();
    return failure();
  }
  for (TaskflowChannelOp channel : channels)
    if (channel && channel->use_empty())
      channel.erase();
  SmallVector<Value> deadIntermediateRoots(producer.getOriginalWriteMemrefs());
  consumer.erase();
  producer.erase();
  for (Value root : deadIntermediateRoots)
    if (root.use_empty())
      if (auto allocation = root.getDefiningOp<memref::AllocOp>())
        allocation.erase();
  return success();
}

static LogicalResult fuseCanonicalPair(ModuleOp module,
                                       StringRef functionName) {
  FailureOr<CanonicalPair> selected = findCanonicalPair(module, functionName);
  if (failed(selected))
    return failure();
  OpBuilder builder(selected->function.getContext());
  inferAndMarkCanonicalRoles(selected->function, builder);
  return fusePair(selected->producer, selected->consumer);
}

static std::string rangeKey(ArrayRef<int64_t> lower, ArrayRef<int64_t> upper) {
  std::string result;
  for (auto [lo, hi] : llvm::zip(lower, upper)) {
    if (!result.empty())
      result += ";";
    result += std::to_string(lo);
    result += ":";
    result += std::to_string(hi);
  }
  return result;
}

static LogicalResult fuseTileLocalPairs(ModuleOp module,
                                        StringRef functionName) {
  FailureOr<CanonicalPair> selected =
      findCanonicalPair(module, functionName, /*requirePair=*/false);
  if (failed(selected))
    return failure();
  func::FuncOp function = selected->function;
  OpBuilder marker(function.getContext());
  inferAndMarkCanonicalRoles(function, marker);
  SmallVector<TaskflowTaskOp> producers = semanticTasks(function, "producer");
  SmallVector<TaskflowTaskOp> consumers = semanticTasks(function, "consumer");
  if (producers.size() < 2 || consumers.size() < 2)
    return reject(
        function,
        "tile-local fusion requires M/N tiled producer and consumer tasks");

  std::map<std::string, TaskflowTaskOp> consumerByRange;
  for (TaskflowTaskOp consumer : consumers) {
    auto lower = counterBounds(consumer, false);
    auto upper = counterBounds(consumer, true);
    if (lower.empty() || upper.empty())
      return reject(consumer, "consumer has no static tile range");
    std::string key = rangeKey(lower, upper);
    if (consumerByRange.count(key))
      return reject(consumer, "multiple consumers own one tile range");
    consumerByRange[key] = consumer;
  }
  SmallVector<std::pair<TaskflowTaskOp, TaskflowTaskOp>> pairs;
  for (TaskflowTaskOp producer : producers) {
    auto lower = counterBounds(producer, false);
    auto upper = counterBounds(producer, true);
    if (lower.empty() || upper.empty())
      return reject(producer, "producer has no static tile range");
    std::string key = rangeKey(lower, upper);
    auto found = consumerByRange.find(key);
    if (found == consumerByRange.end())
      return reject(producer, "producer and consumer tile domains differ");
    pairs.push_back({producer, found->second});
  }
  if (pairs.size() != consumers.size())
    return reject(function,
                  "consumer tile domains are not covered exactly once");

  // The output of each producer tile may currently be hidden behind one or
  // more completion joins created by the M/N tilers.  Rewire the matching
  // consumer tile to the producer's direct state before fusing; then the
  // now-dead intermediate joins are erased.  No numeric operation happens at
  // a completion join.
  for (auto [producer, consumer] : pairs) {
    Value base = producer.getWillWrites().front();
    auto reads = consumer.getWillReads();
    std::optional<unsigned> stateIndex;
    for (unsigned index = 0; index < reads.size(); ++index) {
      Value input = reads[index];
      auto join = input.getDefiningOp<TaskflowJoinOp>();
      if (join && join.getBase() == base) {
        stateIndex = index;
        break;
      }
      if (input == producer.getDoneWrites().front()) {
        stateIndex = index;
        break;
      }
    }
    if (!stateIndex)
      return reject(consumer,
                    "cannot identify the tile-local intermediate state");
    consumer.getWillReadsMutable()
        .slice(*stateIndex, 1)
        .assign(producer.getDoneWrites().front());
  }
  // Removing an outer join can make the inner join that produced one of its
  // tile states dead as well.  Drain this small completion-only chain until
  // every intermediate state has exactly one direct consumer.
  while (true) {
    SmallVector<TaskflowJoinOp> deadJoins;
    function.walk([&](TaskflowJoinOp join) {
      if (join.getJoined().use_empty())
        deadJoins.push_back(join);
    });
    if (deadJoins.empty())
      break;
    for (TaskflowJoinOp join : llvm::reverse(deadJoins))
      join.erase();
  }
  for (auto [producer, consumer] : pairs) {
    if (failed(fusePair(producer, consumer, /*permitJoins=*/true)))
      return failure();
  }
  function->setAttr(kRewriteSchemaAttr, marker.getStringAttr(kRewriteSchema));
  return success();
}

// A completion join is the only legal graph node produced by this file which
// has no compute body.  It carries disjoint Taskflow write states and never
// adds an arithmetic reduction.  Existing joins are accepted and marked so a
// caller can run this pass idempotently after a tiled rewrite.
static LogicalResult materializeCompletionJoin(ModuleOp module,
                                               StringRef functionName) {
  FailureOr<CanonicalPair> selected =
      findCanonicalPair(module, functionName, /*requirePair=*/false);
  if (failed(selected))
    return failure();
  func::FuncOp function = selected->function;
  func::ReturnOp returnOp;
  function.walk([&](func::ReturnOp candidate) {
    if (candidate.getNumOperands() == 1)
      returnOp = candidate;
  });
  if (!returnOp)
    return reject(function,
                  "completion join requires one returned memref state");
  Value returned = returnOp.getOperand(0);
  if (auto existing = returned.getDefiningOp<TaskflowJoinOp>()) {
    OpBuilder builder(existing);
    existing->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
    existing->setAttr(kRewriteSchemaAttr,
                      builder.getStringAttr(kRewriteSchema));
    return success();
  }
  SmallVector<TaskflowTaskOp> terminals;
  function.walk([&](TaskflowTaskOp task) {
    if (task.getDoneWrites().size() == 1 &&
        task.getDoneWrites().front() == returned)
      terminals.push_back(task);
  });
  if (terminals.size() < 2)
    return success();
  Value base = terminals.front().getWillWrites().front();
  auto baseType = dyn_cast<MemRefType>(base.getType());
  if (!baseType || !baseType.hasStaticShape() || baseType.getRank() != 2)
    return reject(function, "completion join requires a static rank-2 memref");
  SmallVector<Value> states;
  SmallVector<int64_t> lower = counterBounds(terminals.front(), false);
  SmallVector<int64_t> upper = counterBounds(terminals.front(), true);
  if (lower.size() != 2 || upper.size() != 2)
    return reject(function,
                  "completion join requires static two-axis tile ranges");
  int64_t axis = -1;
  for (int64_t candidateAxis = 0; candidateAxis < 2; ++candidateAxis) {
    bool compatible = true;
    for (TaskflowTaskOp task : terminals) {
      if (task.getWillWrites().front() != base) {
        compatible = false;
        break;
      }
      auto taskLower = counterBounds(task, false);
      auto taskUpper = counterBounds(task, true);
      if (taskLower.size() != 2 || taskUpper.size() != 2)
        compatible = false;
      for (int64_t dim = 0; compatible && dim < 2; ++dim)
        if (dim != candidateAxis &&
            (taskLower[dim] != lower[dim] || taskUpper[dim] != upper[dim]))
          compatible = false;
    }
    if (compatible) {
      axis = candidateAxis;
      break;
    }
  }
  if (axis < 0)
    return reject(function,
                  "completion states do not form one contiguous rank-2 axis");
  // A join is valid only when the selected intervals exactly cover the
  // parent.  The Taskflow verifier performs the final ordered interval proof.
  lower[axis] = 0;
  upper[axis] = baseType.getShape()[axis];
  for (TaskflowTaskOp task : terminals)
    states.push_back(task.getDoneWrites().front());
  OpBuilder builder(returnOp);
  auto join = builder.create<TaskflowJoinOp>(
      returnOp.getLoc(), returned.getType(), states, base,
      builder.getI64IntegerAttr(axis), builder.getDenseI64ArrayAttr(lower),
      builder.getDenseI64ArrayAttr(upper));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr(kRewriteSchemaAttr, builder.getStringAttr(kRewriteSchema));
  returnOp.setOperand(0, join.getJoined());
  return success();
}

//===----------------------------------------------------------------------===//
// Pass wrappers
//===----------------------------------------------------------------------===//

template <typename Derived>
struct JointModulePass : public PassWrapper<Derived, OperationPass<ModuleOp>> {
  JointModulePass() = default;
  JointModulePass(const JointModulePass &other)
      : PassWrapper<Derived, OperationPass<ModuleOp>>(other) {}

  mlir::Pass::Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Taskflow function name"),
      llvm::cl::init("")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, TaskflowDialect>();
  }
};

struct TileOutputMPass : public JointModulePass<TileOutputMPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileOutputMPass)
  TileOutputMPass() = default;
  TileOutputMPass(const TileOutputMPass &other)
      : JointModulePass<TileOutputMPass>(other) {}
  StringRef getArgument() const override { return "tile-output-m"; }
  StringRef getDescription() const override {
    return "Tile semantic Taskflow outputs along M";
  }
  Option<int64_t> tileSize{*this, "tile-size", llvm::cl::desc("M tile width"),
                           llvm::cl::init(0)};
  void runOnOperation() override {
    if (tileSize < 2 ||
        failed(tileAxis(getOperation(), functionName, 0, tileSize)))
      signalPassFailure();
  }
};

struct TileOutputNPass : public JointModulePass<TileOutputNPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileOutputNPass)
  TileOutputNPass() = default;
  TileOutputNPass(const TileOutputNPass &other)
      : JointModulePass<TileOutputNPass>(other) {}
  StringRef getArgument() const override { return "tile-output-n"; }
  StringRef getDescription() const override {
    return "Tile semantic Taskflow outputs along N";
  }
  Option<int64_t> tileSize{*this, "tile-size", llvm::cl::desc("N tile width"),
                           llvm::cl::init(0)};
  void runOnOperation() override {
    if (tileSize < 2 ||
        failed(tileAxis(getOperation(), functionName, 1, tileSize)))
      signalPassFailure();
  }
};

struct FuseProducerConsumerPass
    : public JointModulePass<FuseProducerConsumerPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseProducerConsumerPass)
  FuseProducerConsumerPass() = default;
  FuseProducerConsumerPass(const FuseProducerConsumerPass &other)
      : JointModulePass<FuseProducerConsumerPass>(other) {}
  StringRef getArgument() const override { return "fuse-producer-consumer"; }
  StringRef getDescription() const override {
    return "Fuse one canonical producer and consumer by SSA forwarding";
  }
  void runOnOperation() override {
    if (failed(fuseCanonicalPair(getOperation(), functionName)))
      signalPassFailure();
  }
};

struct FuseTileLocalPairPass : public JointModulePass<FuseTileLocalPairPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTileLocalPairPass)
  FuseTileLocalPairPass() = default;
  FuseTileLocalPairPass(const FuseTileLocalPairPass &other)
      : JointModulePass<FuseTileLocalPairPass>(other) {}
  StringRef getArgument() const override { return "fuse-tile-local-pair"; }
  StringRef getDescription() const override {
    return "Fuse matching M/N producer and consumer tile pairs";
  }
  void runOnOperation() override {
    if (failed(fuseTileLocalPairs(getOperation(), functionName)))
      signalPassFailure();
  }
};

struct MaterializeCompletionJoinPass
    : public JointModulePass<MaterializeCompletionJoinPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeCompletionJoinPass)
  MaterializeCompletionJoinPass() = default;
  MaterializeCompletionJoinPass(const MaterializeCompletionJoinPass &other)
      : JointModulePass<MaterializeCompletionJoinPass>(other) {}
  StringRef getArgument() const override {
    return "materialize-completion-join";
  }
  StringRef getDescription() const override {
    return "Materialize a non-compute completion join for disjoint states";
  }
  void runOnOperation() override {
    if (failed(materializeCompletionJoin(getOperation(), functionName)))
      signalPassFailure();
  }
};

// Compatibility entry point for clients which have not yet split their pass
// pipeline.  New callers should use one of the independent pass factories
// above.  The action syntax is a comma separated list of
// `TileOutputM[:width]`, `TileOutputN[:width]`, `FuseProducerConsumer`,
// `FuseTileLocalPair`, or `MaterializeCompletionJoin`.
struct MaterializeJointSemanticRewritePass
    : public JointModulePass<MaterializeJointSemanticRewritePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeJointSemanticRewritePass)
  MaterializeJointSemanticRewritePass() = default;
  MaterializeJointSemanticRewritePass(
      const MaterializeJointSemanticRewritePass &other)
      : JointModulePass<MaterializeJointSemanticRewritePass>(other) {}
  StringRef getArgument() const override {
    return "materialize-joint-semantic-rewrite";
  }
  StringRef getDescription() const override {
    return "Apply a selected hardware-independent semantic Taskflow rewrite";
  }
  Option<std::string> action{*this, "action",
                             llvm::cl::desc("Comma separated semantic actions"),
                             llvm::cl::init("")};
  Option<int64_t> tileSize{*this, "tile-size",
                           llvm::cl::desc("Default M/N tile width"),
                           llvm::cl::init(0)};
  void runOnOperation() override {
    if (action.empty()) {
      getOperation().emitError() << "action is required; use independent "
                                    "semantic rewrite passes for enumeration";
      return signalPassFailure();
    }
    SmallVector<StringRef> actions;
    StringRef(action.getValue()).split(actions, ',');
    for (StringRef raw : actions) {
      SmallVector<StringRef> fields;
      raw.split(fields, ':');
      StringRef kind = fields.front().trim();
      int64_t width = tileSize;
      if (fields.size() == 2 && fields[1].getAsInteger(10, width)) {
        getOperation().emitError()
            << "invalid tile width in action '" << raw << "'";
        return signalPassFailure();
      }
      LogicalResult result = success();
      if (kind == "TileOutputM")
        result = tileAxis(getOperation(), functionName, 0, width);
      else if (kind == "TileOutputN")
        result = tileAxis(getOperation(), functionName, 1, width);
      else if (kind == "FuseProducerConsumer")
        result = fuseCanonicalPair(getOperation(), functionName);
      else if (kind == "FuseTileLocalPair")
        result = fuseTileLocalPairs(getOperation(), functionName);
      else if (kind == "MaterializeCompletionJoin")
        result = materializeCompletionJoin(getOperation(), functionName);
      else {
        getOperation().emitError()
            << "unsupported semantic action '" << kind << "'";
        result = failure();
      }
      if (failed(result))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createTileOutputMPass() {
  return std::make_unique<TileOutputMPass>();
}
std::unique_ptr<Pass> createTileOutputNPass() {
  return std::make_unique<TileOutputNPass>();
}
std::unique_ptr<Pass> createFuseProducerConsumerPass() {
  return std::make_unique<FuseProducerConsumerPass>();
}
std::unique_ptr<Pass> createFuseTileLocalPairPass() {
  return std::make_unique<FuseTileLocalPairPass>();
}
std::unique_ptr<Pass> createMaterializeCompletionJoinPass() {
  return std::make_unique<MaterializeCompletionJoinPass>();
}
std::unique_ptr<Pass> createMaterializeJointSemanticRewritePass() {
  return std::make_unique<MaterializeJointSemanticRewritePass>();
}

} // namespace mlir::amoeba::neura
