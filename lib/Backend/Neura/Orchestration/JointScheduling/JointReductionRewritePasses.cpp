//===- JointReductionRewritePasses.cpp ------------------------*- C++ -*-===//
//
// Hardware independent K-reduction rewrites for the ORBIT semantic graph.
//
// Every operation in this file is a separate ModuleOp pass.  The passes
// intentionally materialize ordinary Taskflow SSA states: a sequential K
// rewrite creates a state chain, while a parallel rewrite creates private
// partial states followed by an explicit numeric reduction.  Completion
// joins are not used as arithmetic reductions.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kSemanticSchemaAttr = "amoeba.semantic.schema";
constexpr StringLiteral kSemanticKindAttr = "amoeba.semantic.task_kind";
constexpr StringLiteral kSemanticRoleAttr = "amoeba.semantic.task_role";
constexpr StringLiteral kSemanticKPolicyAttr = "amoeba.semantic.k_policy";
constexpr StringLiteral kSemanticTopologyAttr =
    "amoeba.semantic.reduction_topology";
constexpr StringLiteral kSemanticKRangeAttr = "amoeba.semantic.K_range";
constexpr StringLiteral kSemanticMRangeAttr = "amoeba.semantic.M_range";
constexpr StringLiteral kSemanticNRangeAttr = "amoeba.semantic.N_range";
constexpr StringLiteral kSemanticInitialCOwnershipAttr =
    "amoeba.semantic.initial_C_ownership";
constexpr StringLiteral kSemanticFinalCOwnershipAttr =
    "amoeba.semantic.final_C_ownership";
constexpr StringLiteral kSemanticNumericReductionAttr =
    "amoeba.semantic.numeric_reduction";
constexpr StringLiteral kSemanticReductionFinalAttr =
    "amoeba.semantic.reduction_final";
constexpr StringLiteral kSemanticPrivatePartialAttr =
    "amoeba.semantic.private_partial";
constexpr StringLiteral kSemanticFusionLayerAttr =
    "amoeba.semantic.fusion_layer";
constexpr StringLiteral kSemanticFusionModeAttr = "amoeba.semantic.fusion_mode";
constexpr StringLiteral kSemanticRewriteSchemaAttr =
    "amoeba.semantic.rewrite_schema";
constexpr StringLiteral kSemanticRewriteSchema =
    "orbit-joint-semantic-rewrite-v1";
constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";

// The typed graph facts pass consumes these attributes when it proves that a
// Taskflow edge moves only the producer/consumer tile.  They are deliberately
// emitted by the rewrite itself, rather than inferred from task names.
constexpr StringLiteral kInputRegionLowersAttr =
    "amoeba.tiling.input_region_lowers";
constexpr StringLiteral kInputRegionUppersAttr =
    "amoeba.tiling.input_region_uppers";
constexpr StringLiteral kInputRegionReasonsAttr =
    "amoeba.tiling.input_region_reasons";
constexpr StringLiteral kOutputRegionLowersAttr =
    "amoeba.tiling.output_region_lowers";
constexpr StringLiteral kOutputRegionUppersAttr =
    "amoeba.tiling.output_region_uppers";

struct StaticRegion {
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

struct SourcePipeline {
  Location loc;
  explicit SourcePipeline(Location location) : loc(location) {}
  func::FuncOp function;
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  Value A;
  Value B;
  Value C;
  Value T;
  Value E;
  Value D;
  MemRefType tensorType;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  scf::ForOp reductionLoop;
};

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

static bool isStructuralAttribute(StringRef name) {
  return name == "task_name" || name == "operandSegmentSizes" ||
         name == "resultSegmentSizes" || name == "operand_segment_sizes" ||
         name == "result_segment_sizes";
}

static StringRef taskKind(TaskflowTaskOp task) {
  if (auto attr = task->getAttrOfType<StringAttr>(kSemanticKindAttr))
    return attr.getValue();
  if (task.getTaskName().starts_with("consumer"))
    return "elementwise_consumer";
  if (task.getTaskName().starts_with("reduction"))
    return "reduction";
  return "gemm_producer";
}

static StringRef taskRole(TaskflowTaskOp task) {
  if (auto attr = task->getAttrOfType<StringAttr>(kSemanticRoleAttr))
    return attr.getValue();
  StringRef kind = taskKind(task);
  if (kind == "elementwise_consumer")
    return "consumer";
  if (kind == "reduction")
    return "reduction";
  return "producer";
}

static bool isProducer(TaskflowTaskOp task) {
  return taskRole(task) == "producer" || taskKind(task) == "gemm_producer" ||
         taskKind(task) == "fused_producer_consumer";
}

static bool isConsumer(TaskflowTaskOp task) {
  return taskRole(task) == "consumer" ||
         taskKind(task) == "elementwise_consumer";
}

static FailureOr<func::FuncOp> selectFunction(ModuleOp module,
                                              StringRef functionName) {
  SmallVector<func::FuncOp> functions;
  module.walk([&](func::FuncOp function) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask &&
        (functionName.empty() || function.getSymName() == functionName))
      functions.push_back(function);
  });
  if (functions.size() != 1) {
    module.emitError() << "joint K rewrite requires exactly one Taskflow "
                          "function (found "
                       << functions.size() << ")";
    return failure();
  }
  return functions.front();
}

static FailureOr<TaskflowTaskOp> selectUniqueTask(func::FuncOp function,
                                                  bool producer) {
  SmallVector<TaskflowTaskOp> matches;
  function.walk([&](TaskflowTaskOp task) {
    // Parallel K tiling leaves the original producer as the pending owner of
    // T until the independent materialization pass publishes its reduction.
    // Private partial producers are implementation states, not another
    // source producer for the pipeline recognizer.
    if (producer && task->hasAttr(kSemanticPrivatePartialAttr))
      return;
    if ((producer && isProducer(task)) || (!producer && isConsumer(task)))
      matches.push_back(task);
  });
  if (matches.size() != 1) {
    function.emitError() << "joint K rewrite requires exactly one "
                         << (producer ? "producer" : "consumer")
                         << " task (found " << matches.size() << ")";
    return failure();
  }
  return matches.front();
}

static SmallVector<TaskflowCounterOp> counterChain(TaskflowTaskOp task) {
  SmallVector<TaskflowCounterOp> counters;
  if (!task || !task.getBody().hasOneBlock())
    return counters;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
  return counters;
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

static LogicalResult validateStaticTensor(Value value, StringRef name,
                                          MemRefType expected) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      type.getRank() != 2 || !type.getElementType().isInteger(32))
    return failure();
  if (expected && type != expected)
    return failure();
  return success();
}

static bool hasTaskflowJoin(func::FuncOp function) {
  bool found = false;
  function.walk([&](Operation *operation) {
    found |= operation->getName().getStringRef() == "taskflow.join";
  });
  return found;
}

static bool hasOneDirectUse(Value value, Operation *owner,
                            unsigned operandIndex) {
  unsigned count = 0;
  for (OpOperand &use : value.getUses()) {
    if (use.getOwner() == owner && use.getOperandNumber() == operandIndex)
      ++count;
    else
      return false;
  }
  return count == 1;
}

static FailureOr<SourcePipeline> analyzeSource(ModuleOp module,
                                               StringRef functionName) {
  FailureOr<func::FuncOp> function = selectFunction(module, functionName);
  if (failed(function))
    return failure();
  FailureOr<TaskflowTaskOp> producer = selectUniqueTask(*function, true);
  FailureOr<TaskflowTaskOp> consumer = selectUniqueTask(*function, false);
  if (failed(producer) || failed(consumer))
    return failure();
  if (hasTaskflowJoin(*function))
    return reject(*function,
                  "K reduction rewrites require direct task states; completion "
                  "joins must be materialized after K reduction");
  if (producer->getOperation()->getBlock() !=
          consumer->getOperation()->getBlock() ||
      !producer->getOperation()->isBeforeInBlock(consumer->getOperation()))
    return reject(*consumer, "producer must precede its direct consumer");

  if (producer->getWillReads().size() != 3 ||
      producer->getWillWrites().size() != 1 ||
      producer->getDoneReads().size() != 0 ||
      producer->getDoneWrites().size() != 1 ||
      producer->getValueInputs().size() != 0 ||
      producer->getOriginalReadMemrefs().size() != 3 ||
      producer->getOriginalWriteMemrefs().size() != 1 ||
      producer->getBody().front().getNumArguments() !=
          producer->getWillReads().size() + producer->getWillWrites().size() +
              producer->getValueInputs().size())
    return reject(*producer, "K rewrite expects one A/B/C input state and one "
                             "producer output state");
  if (consumer->getWillReads().size() != 2 ||
      consumer->getWillWrites().size() != 1 ||
      consumer->getDoneReads().size() != 0 ||
      consumer->getDoneWrites().size() != 1 ||
      consumer->getValueInputs().size() != 0 ||
      consumer->getOriginalReadMemrefs().size() != 2 ||
      consumer->getOriginalWriteMemrefs().size() != 1)
    return reject(*consumer,
                  "K rewrite expects one T and one E consumer input");

  Value produced = producer->getDoneWrites().front();
  unsigned consumerInput = 0;
  for (; consumerInput < consumer->getWillReads().size(); ++consumerInput)
    if (consumer->getWillReads()[consumerInput] == produced)
      break;
  if (consumerInput == consumer->getWillReads().size())
    return reject(*consumer, "consumer does not directly read producer output");
  if (!hasOneDirectUse(produced, consumer->getOperation(), consumerInput))
    return reject(*producer,
                  "producer output has an external or ambiguous user");

  SourcePipeline result(producer->getOperation()->getLoc());
  result.function = *function;
  result.producer = *producer;
  result.consumer = *consumer;
  result.A = producer->getWillReads()[0];
  result.B = producer->getWillReads()[1];
  result.C = producer->getWillReads()[2];
  result.T = producer->getWillWrites()[0];
  result.E = consumer->getWillReads()[consumerInput == 0 ? 1 : 0];
  result.D = consumer->getWillWrites().front();
  result.tensorType = dyn_cast<MemRefType>(result.T.getType());
  if (!result.tensorType ||
      failed(validateStaticTensor(result.A, "A", result.tensorType)) ||
      failed(validateStaticTensor(result.B, "B", result.tensorType)) ||
      failed(validateStaticTensor(result.C, "C_initial", result.tensorType)) ||
      failed(validateStaticTensor(result.T, "T", result.tensorType)) ||
      failed(validateStaticTensor(result.E, "E", result.tensorType)) ||
      failed(validateStaticTensor(result.D, "D", result.tensorType)))
    return reject(*producer,
                  "canonical K rewrite requires static 2-D i32 tensors");
  result.M = result.tensorType.getShape()[0];
  result.N = result.tensorType.getShape()[1];

  result.counters = counterChain(*producer);
  result.hyperblock = findHyperblock(*producer);
  if (result.counters.size() != 2 || !result.hyperblock ||
      result.hyperblock.getIndices().size() != 2 ||
      result.hyperblock.getIterArgs().size() != 0 ||
      result.hyperblock.getNumResults() != 0 ||
      result.hyperblock.getBody().empty())
    return reject(*producer, "producer must contain one static M/N "
                             "counter chain and one hyperblock");
  for (size_t i = 0; i < result.counters.size(); ++i) {
    auto lower = constantIndex(result.counters[i].getLowerBound());
    auto upper = constantIndex(result.counters[i].getUpperBound());
    auto step = constantIndex(result.counters[i].getStep());
    bool parentMismatch =
        i == 0 ? static_cast<bool>(result.counters[i].getParentIndex())
               : result.counters[i].getParentIndex() !=
                     result.counters[i - 1].getCounterIndex();
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower ||
        parentMismatch ||
        result.hyperblock.getIndices()[i] !=
            result.counters[i].getCounterIndex())
      return reject(*producer, "producer counters must be constant unit-step "
                               "M/N chain");
    if (i == 0)
      result.M = *upper - *lower;
    else
      result.N = *upper - *lower;
  }

  SmallVector<scf::ForOp> loops;
  result.hyperblock.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1)
    return reject(result.hyperblock,
                  "producer must contain exactly one inner K reduction loop");
  result.reductionLoop = loops.front();
  auto kLower = constantIndex(result.reductionLoop.getLowerBound());
  auto kUpper = constantIndex(result.reductionLoop.getUpperBound());
  auto kStep = constantIndex(result.reductionLoop.getStep());
  if (!kLower || !kUpper || !kStep || *kStep != 1 || *kUpper <= *kLower ||
      result.reductionLoop.getNumRegionIterArgs() != 1 ||
      !result.reductionLoop.getResult(0).getType().isInteger(32))
    return reject(result.reductionLoop,
                  "K reduction loop must have one static i32 accumulator");
  result.K = *kUpper - *kLower;

  if (!provesDistinctStorage(result.A, result.B) ||
      !provesDistinctStorage(result.A, result.C) ||
      !provesDistinctStorage(result.A, result.T) ||
      !provesDistinctStorage(result.A, result.E) ||
      !provesDistinctStorage(result.A, result.D))
    return reject(*producer,
                  "K rewrite requires explicit noalias or allocation roots");
  if (result.producer->hasAttr("amoeba.semantic.k_tiled"))
    return reject(*producer, "producer K reduction is already tiled");
  return result;
}

static StaticRegion fullRegion(const SourcePipeline &source) {
  return {{0, 0}, {source.M, source.N}};
}

static StaticRegion aRegion(const SourcePipeline &source, int64_t lower,
                            int64_t upper) {
  return {{0, lower}, {source.M, upper}};
}

static StaticRegion bRegion(const SourcePipeline &source, int64_t lower,
                            int64_t upper) {
  return {{lower, 0}, {upper, source.N}};
}

static void setRegionContracts(OpBuilder &builder, TaskflowTaskOp task,
                               ArrayRef<StaticRegion> inputRegions,
                               ArrayRef<StaticRegion> outputRegions,
                               StringRef reason = "semantic_k_tile") {
  SmallVector<Attribute> lowers;
  SmallVector<Attribute> uppers;
  SmallVector<Attribute> reasons;
  for (const StaticRegion &region : inputRegions) {
    lowers.push_back(builder.getDenseI64ArrayAttr(region.lower));
    uppers.push_back(builder.getDenseI64ArrayAttr(region.upper));
    reasons.push_back(builder.getStringAttr(reason));
  }
  task->setAttr(kInputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kInputRegionUppersAttr, builder.getArrayAttr(uppers));
  task->setAttr(kInputRegionReasonsAttr, builder.getArrayAttr(reasons));
  lowers.clear();
  uppers.clear();
  for (const StaticRegion &region : outputRegions) {
    lowers.push_back(builder.getDenseI64ArrayAttr(region.lower));
    uppers.push_back(builder.getDenseI64ArrayAttr(region.upper));
  }
  task->setAttr(kOutputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kOutputRegionUppersAttr, builder.getArrayAttr(uppers));
}

static void
setIncomingEdges(OpBuilder &builder, TaskflowTaskOp task,
                 ArrayRef<std::tuple<StringRef, StringRef, StringRef>> edges) {
  SmallVector<Attribute> attrs;
  for (auto [source, role, scope] : edges)
    attrs.push_back(builder.getStringAttr(
        (Twine(source) + "|" + role + "|" + scope).str()));
  task->setAttr(kSemanticIncomingEdgesAttr, builder.getArrayAttr(attrs));
}

static void copyCustomAttributes(TaskflowTaskOp source, TaskflowTaskOp target) {
  for (NamedAttribute attr : source->getAttrs())
    if (!isStructuralAttribute(attr.getName().strref()))
      target->setAttr(attr.getName(), attr.getValue());
}

static void markTask(OpBuilder &builder, TaskflowTaskOp task, StringRef kind,
                     StringRef role, StringRef policy, StringRef topology) {
  task->setAttr(
      kSemanticSchemaAttr,
      builder.getStringAttr("orbit-joint-taskflow-materialization-v1"));
  task->setAttr(kSemanticKindAttr, builder.getStringAttr(kind));
  task->setAttr(kSemanticRoleAttr, builder.getStringAttr(role));
  task->setAttr(kSemanticKPolicyAttr, builder.getStringAttr(policy));
  task->setAttr(kSemanticTopologyAttr, builder.getStringAttr(topology));
  task->setAttr(kSemanticRewriteSchemaAttr,
                builder.getStringAttr(kSemanticRewriteSchema));
}

static void setRangeAttrs(OpBuilder &builder, TaskflowTaskOp task,
                          ArrayRef<int64_t> mRange, ArrayRef<int64_t> nRange,
                          ArrayRef<int64_t> kRange) {
  task->setAttr(kSemanticMRangeAttr, builder.getDenseI64ArrayAttr(mRange));
  task->setAttr(kSemanticNRangeAttr, builder.getDenseI64ArrayAttr(nRange));
  task->setAttr(kSemanticKRangeAttr, builder.getDenseI64ArrayAttr(kRange));
}

static void addTaskYield(Block *body, Value output) {
  OpBuilder builder = OpBuilder::atBlockEnd(body);
  builder.create<TaskflowYieldOp>(output.getLoc(), ValueRange{},
                                  ValueRange{output}, ValueRange{});
}

static void createCounterConstants(OpBuilder &builder, Location loc,
                                   int64_t lower, int64_t upper, Value &lb,
                                   Value &ub, Value &step) {
  lb = builder.create<arith::ConstantIndexOp>(loc, lower);
  ub = builder.create<arith::ConstantIndexOp>(loc, upper);
  step = builder.create<arith::ConstantIndexOp>(loc, 1);
}

static SmallVector<TaskflowCounterOp>
createMNcounters(OpBuilder &builder, Location loc, int64_t M, int64_t N) {
  Value mLower, mUpper, mStep;
  Value nLower, nUpper, nStep;
  createCounterConstants(builder, loc, 0, M, mLower, mUpper, mStep);
  auto m = builder.create<TaskflowCounterOp>(
      loc, builder.getIndexType(), Value{}, mLower, mUpper, mStep, StringAttr{},
      StringAttr{}, IntegerAttr{});
  createCounterConstants(builder, loc, 0, N, nLower, nUpper, nStep);
  auto n = builder.create<TaskflowCounterOp>(
      loc, builder.getIndexType(), m.getCounterIndex(), nLower, nUpper, nStep,
      StringAttr{}, StringAttr{}, IntegerAttr{});
  return {m, n};
}

static LogicalResult addReductionBody(OpBuilder &taskBuilder, Location loc,
                                      ArrayRef<Value> readArgs, Value outputArg,
                                      int64_t M, int64_t N,
                                      bool includeInitialC, int64_t kLower = 0,
                                      int64_t kUpper = 0, Value A = Value{},
                                      Value B = Value{}) {
  auto counters = createMNcounters(taskBuilder, loc, M, N);
  auto hyperblock = taskBuilder.create<TaskflowHyperblockOp>(
      loc, TypeRange{},
      ValueRange{counters[0].getCounterIndex(), counters[1].getCounterIndex()},
      ValueRange{});
  Block *hbBody = &hyperblock.getBody().emplaceBlock();
  hbBody->addArguments({taskBuilder.getIndexType(), taskBuilder.getIndexType()},
                       {loc, loc});
  OpBuilder hbBuilder = OpBuilder::atBlockEnd(hbBody);
  Value m = hbBody->getArgument(0);
  Value n = hbBody->getArgument(1);

  if (A && B) {
    Value kLb, kUb, kStep;
    // Keep the K bounds in the hyperblock region.  The task body is isolated
    // from its parent, and the loop is nested below the hyperblock; placing
    // these values at the loop's immediate region level also keeps dominance
    // explicit for freshly built task bodies.
    createCounterConstants(hbBuilder, loc, kLower, kUpper, kLb, kUb, kStep);
    auto zero = hbBuilder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto loop =
        hbBuilder.create<scf::ForOp>(loc, kLb, kUb, kStep, ValueRange{zero});
    Block *loopBody = loop.getBody();
    // The SCF builder leaves the loop body without a terminator.  Build the
    // body at its end and append the yield after the reduction operations.
    OpBuilder loopBuilder = OpBuilder::atBlockEnd(loopBody);
    Value a = loopBuilder.create<memref::LoadOp>(
        loc, A, ValueRange{m, loop.getInductionVar()});
    Value b = loopBuilder.create<memref::LoadOp>(
        loc, B, ValueRange{loop.getInductionVar(), n});
    Value product = loopBuilder.create<arith::MulIOp>(loc, a, b);
    Value sum = loopBuilder.create<arith::AddIOp>(
        loc, loop.getRegionIterArgs().front(), product);
    loopBuilder.create<scf::YieldOp>(loc, sum);
    hbBuilder.setInsertionPointAfter(loop);
    hbBuilder.create<memref::StoreOp>(loc, loop.getResult(0), outputArg,
                                      ValueRange{m, n});
  } else {
    unsigned first = includeInitialC ? 0 : 0;
    if (readArgs.empty() || (!includeInitialC && readArgs.empty()))
      return failure();
    Value accumulator;
    if (includeInitialC)
      accumulator = hbBuilder.create<memref::LoadOp>(loc, readArgs[first++],
                                                     ValueRange{m, n});
    else
      accumulator = hbBuilder.create<arith::ConstantIntOp>(loc, 0, 32);
    for (unsigned index = first; index < readArgs.size(); ++index) {
      Value value = hbBuilder.create<memref::LoadOp>(loc, readArgs[index],
                                                     ValueRange{m, n});
      accumulator = hbBuilder.create<arith::AddIOp>(loc, accumulator, value);
    }
    hbBuilder.create<memref::StoreOp>(loc, accumulator, outputArg,
                                      ValueRange{m, n});
  }
  OpBuilder endBuilder = OpBuilder::atBlockEnd(hbBody);
  endBuilder.create<TaskflowHyperblockYieldOp>(loc);
  return success();
}

static FailureOr<TaskflowTaskOp>
buildPartialProducer(OpBuilder &builder, const SourcePipeline &source,
                     Value partial, StringRef name, int64_t kLower,
                     int64_t kUpper, unsigned index, StringRef topology) {
  SmallVector<Value> reads{source.A, source.B};
  SmallVector<Value> roots{source.A, source.B};
  auto task = builder.create<TaskflowTaskOp>(
      source.loc, TypeRange{}, TypeRange{partial.getType()}, TypeRange{}, reads,
      ValueRange{partial}, ValueRange{}, builder.getStringAttr(name), roots,
      ValueRange{partial});
  Block *body = &task.getBody().emplaceBlock();
  body->addArguments(
      {source.A.getType(), source.B.getType(), partial.getType()},
      {source.loc, source.loc, source.loc});
  OpBuilder taskBuilder = OpBuilder::atBlockEnd(body);
  if (failed(addReductionBody(
          taskBuilder, source.loc, {body->getArgument(0), body->getArgument(1)},
          body->getArgument(2), source.M, source.N, false, kLower, kUpper,
          body->getArgument(0), body->getArgument(1))))
    return failure();
  addTaskYield(body, body->getArgument(2));
  markTask(builder, task, "gemm_producer", "producer", "parallel", topology);
  task->setAttr(kSemanticPrivatePartialAttr, builder.getUnitAttr());
  task->setAttr("amoeba.semantic.output_visibility",
                builder.getStringAttr("private_partial"));
  task->setAttr("amoeba.semantic.logical_k_block",
                builder.getI64IntegerAttr(index));
  task->setAttr("amoeba.semantic.partial_buffer", builder.getStringAttr(name));
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N}, {kLower, kUpper});
  setRegionContracts(
      builder, task,
      {aRegion(source, kLower, kUpper), bRegion(source, kLower, kUpper)},
      {fullRegion(source)}, "private_partial");
  setIncomingEdges(builder, task, {});
  return task;
}

static FailureOr<TaskflowTaskOp> buildNumericReduction(
    OpBuilder &builder, const SourcePipeline &source,
    ArrayRef<Value> partialStates, ArrayRef<Value> partialRoots, StringRef name,
    StringRef topology, Value outputRoot, bool includeInitialC, bool fused,
    bool finalReduction,
    ArrayRef<std::tuple<StringRef, StringRef, StringRef>> incoming) {
  SmallVector<Value> reads;
  SmallVector<Value> roots;
  if (includeInitialC) {
    reads.push_back(source.C);
    roots.push_back(source.C);
  }
  reads.append(partialStates.begin(), partialStates.end());
  roots.append(partialRoots.begin(), partialRoots.end());
  auto task = builder.create<TaskflowTaskOp>(
      source.loc, TypeRange{}, TypeRange{outputRoot.getType()}, TypeRange{},
      reads, ValueRange{outputRoot}, ValueRange{}, builder.getStringAttr(name),
      roots, ValueRange{outputRoot});
  Block *body = &task.getBody().emplaceBlock();
  SmallVector<Type> bodyTypes;
  for (Value value : reads)
    bodyTypes.push_back(value.getType());
  bodyTypes.push_back(outputRoot.getType());
  SmallVector<Location> locations(bodyTypes.size(), source.loc);
  body->addArguments(bodyTypes, locations);
  SmallVector<Value> readArgs;
  for (unsigned index = 0; index < reads.size(); ++index)
    readArgs.push_back(body->getArgument(index));
  OpBuilder taskBuilder = OpBuilder::atBlockEnd(body);
  if (failed(addReductionBody(taskBuilder, source.loc, readArgs,
                              body->getArgument(reads.size()), source.M,
                              source.N, includeInitialC)))
    return failure();
  addTaskYield(body, body->getArgument(reads.size()));
  markTask(builder, task, fused ? "fused_reduction_consumer" : "reduction",
           "reduction", "parallel", topology);
  task->setAttr(kSemanticNumericReductionAttr, builder.getUnitAttr());
  if (finalReduction)
    task->setAttr(kSemanticReductionFinalAttr, builder.getUnitAttr());
  else
    task->setAttr(kSemanticPrivatePartialAttr, builder.getUnitAttr());
  task->setAttr("amoeba.semantic.output_visibility",
                fused ? builder.getStringAttr("fused_internal")
                      : builder.getStringAttr("global_materialized"));
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N}, {0, source.K});
  SmallVector<StaticRegion> inputRegions;
  inputRegions.push_back(fullRegion(source));
  inputRegions.resize(reads.size(), fullRegion(source));
  setRegionContracts(builder, task, inputRegions, {fullRegion(source)},
                     "numeric_reduction");
  setIncomingEdges(builder, task, incoming);
  return task;
}

static void markConsumerContract(OpBuilder &builder, TaskflowTaskOp consumer,
                                 const SourcePipeline &source,
                                 StringRef inputSource, StringRef role,
                                 StringRef policy = "sequential",
                                 StringRef topology = "none") {
  markTask(builder, consumer, "elementwise_consumer", "consumer", policy,
           topology);
  setRangeAttrs(builder, consumer, {0, source.M}, {0, source.N}, {0, source.K});
  setRegionContracts(builder, consumer,
                     {fullRegion(source), fullRegion(source)},
                     {fullRegion(source)}, "consumer_tile");
  setIncomingEdges(builder, consumer, {{inputSource, role, "tile_local"}});
}

static std::optional<unsigned> findInput(TaskflowTaskOp task, Value value) {
  for (auto indexed : llvm::enumerate(task.getWillReads()))
    if (indexed.value() == value)
      return indexed.index();
  return std::nullopt;
}

static void setFunctionRewriteAttrs(OpBuilder &builder, func::FuncOp function,
                                    StringRef policy, StringRef topology) {
  function->setAttr(kSemanticRewriteSchemaAttr,
                    builder.getStringAttr(kSemanticRewriteSchema));
  function->setAttr(kSemanticKPolicyAttr, builder.getStringAttr(policy));
  function->setAttr(kSemanticTopologyAttr, builder.getStringAttr(topology));
}

static std::optional<StaticRegion> getInputRegion(TaskflowTaskOp task,
                                                  unsigned index) {
  auto lowers = task->getAttrOfType<ArrayAttr>(kInputRegionLowersAttr);
  auto uppers = task->getAttrOfType<ArrayAttr>(kInputRegionUppersAttr);
  if (!lowers || !uppers || index >= lowers.size() || index >= uppers.size())
    return std::nullopt;
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[index]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[index]);
  if (!lower || !upper)
    return std::nullopt;
  return StaticRegion{SmallVector<int64_t>(lower.asArrayRef()),
                      SmallVector<int64_t>(upper.asArrayRef())};
}

static std::optional<StaticRegion> getOutputRegion(TaskflowTaskOp task,
                                                   unsigned index) {
  auto lowers = task->getAttrOfType<ArrayAttr>(kOutputRegionLowersAttr);
  auto uppers = task->getAttrOfType<ArrayAttr>(kOutputRegionUppersAttr);
  if (!lowers || !uppers || index >= lowers.size() || index >= uppers.size())
    return std::nullopt;
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[index]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[index]);
  if (!lower || !upper)
    return std::nullopt;
  return StaticRegion{SmallVector<int64_t>(lower.asArrayRef()),
                      SmallVector<int64_t>(upper.asArrayRef())};
}

static StaticRegion fullRegionForValue(Value value) {
  auto type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasStaticShape())
    return {};
  SmallVector<int64_t> lower(type.getRank(), 0);
  SmallVector<int64_t> upper(type.getShape().begin(), type.getShape().end());
  return {std::move(lower), std::move(upper)};
}

struct FusionPair {
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  unsigned consumerInput = 0;
  bool reduction = false;
};

static FailureOr<FusionPair> findReductionFusionPair(ModuleOp module,
                                                     StringRef functionName) {
  FailureOr<func::FuncOp> function = selectFunction(module, functionName);
  if (failed(function))
    return failure();
  SmallVector<TaskflowTaskOp> consumers;
  function->walk([&](TaskflowTaskOp task) {
    if (isConsumer(task))
      consumers.push_back(task);
  });
  if (consumers.size() != 1)
    return reject(*function,
                  "reduction-consumer fusion requires exactly one consumer");
  TaskflowTaskOp consumer = consumers.front();
  SmallVector<FusionPair> pairs;
  for (TaskflowTaskOp task : function->getOps<TaskflowTaskOp>()) {
    StringRef kind = taskKind(task);
    bool reduction = kind == "reduction";
    bool sequentialProducer = kind == "gemm_producer";
    if (!reduction && !sequentialProducer)
      continue;
    for (auto indexed : llvm::enumerate(consumer.getWillReads())) {
      if (indexed.value() != task.getDoneWrites().front())
        continue;
      if (!hasOneDirectUse(task.getDoneWrites().front(),
                           consumer.getOperation(), indexed.index()))
        return reject(task, "fusion requires an exclusive final state user");
      if (reduction && !task->hasAttr(kSemanticReductionFinalAttr))
        return reject(
            task, "partial reduction task cannot fuse before its final task");
      if (sequentialProducer) {
        auto ownership =
            task->getAttrOfType<StringAttr>(kSemanticFinalCOwnershipAttr);
        if (!task->hasAttr("amoeba.semantic.k_tiled") || !ownership ||
            ownership.getValue() != "external_final_C")
          return reject(task, "sequential consumer fusion requires a completed "
                              "final K producer");
      }
      pairs.push_back(
          {task, consumer, static_cast<unsigned>(indexed.index()), reduction});
    }
  }
  if (pairs.size() != 1)
    return reject(consumer,
                  "fusion requires one direct final producer or reduction");
  FusionPair pair = pairs.front();
  if (pair.producer.getOperation()->getBlock() !=
          pair.consumer.getOperation()->getBlock() ||
      !pair.producer.getOperation()->isBeforeInBlock(
          pair.consumer.getOperation()))
    return reject(pair.consumer, "producer and consumer are not ordered");
  for (Operation *operation = pair.producer.getOperation()->getNextNode();
       operation && operation != pair.consumer.getOperation();
       operation = operation->getNextNode()) {
    if (operation->getName().getStringRef() != "taskflow.channel")
      return reject(pair.consumer,
                    "fusion rejects intervening tasks and partial consumers");
  }
  return pair;
}

static FailureOr<TaskflowTaskOp> buildFusedFinalTask(OpBuilder &builder,
                                                     const FusionPair &pair) {
  TaskflowTaskOp producer = pair.producer;
  TaskflowTaskOp consumer = pair.consumer;
  SmallVector<Value> reads;
  SmallVector<Value> roots;
  SmallVector<StaticRegion> inputRegions;
  auto appendRead = [&](Value value, Value root,
                        std::optional<StaticRegion> region) -> LogicalResult {
    if (llvm::is_contained(reads, value))
      return reject(consumer, "fusion would duplicate a task input state");
    reads.push_back(value);
    roots.push_back(root);
    inputRegions.push_back(region ? *region : fullRegionForValue(root));
    if (inputRegions.back().lower.empty() || inputRegions.back().upper.empty())
      return reject(consumer, "fusion input has no static region proof");
    return success();
  };

  for (auto indexed : llvm::enumerate(producer.getWillReads())) {
    std::optional<StaticRegion> region =
        getInputRegion(producer, indexed.index());
    if (failed(appendRead(indexed.value(),
                          producer.getOriginalReadMemrefs()[indexed.index()],
                          region)))
      return failure();
  }
  for (auto indexed : llvm::enumerate(consumer.getWillReads())) {
    if (indexed.index() == pair.consumerInput)
      continue;
    std::optional<StaticRegion> region =
        getInputRegion(consumer, indexed.index());
    if (failed(appendRead(indexed.value(),
                          consumer.getOriginalReadMemrefs()[indexed.index()],
                          region)))
      return failure();
  }
  if (producer.getWillWrites().size() != 1 ||
      consumer.getWillWrites().size() != 1 ||
      consumer.getOriginalWriteMemrefs().size() != 1)
    return reject(consumer, "fusion requires one producer and one D output");

  Value output = consumer.getWillWrites().front();
  Value outputRoot = consumer.getOriginalWriteMemrefs().front();
  StaticRegion outputRegion =
      getOutputRegion(consumer, 0).value_or(fullRegionForValue(outputRoot));
  auto fused = builder.create<TaskflowTaskOp>(
      consumer.getLoc(), TypeRange{}, TypeRange{output.getType()}, TypeRange{},
      reads, ValueRange{output}, ValueRange{},
      builder.getStringAttr((Twine("fused.") + producer.getTaskName() + "." +
                             consumer.getTaskName())
                                .str()),
      roots, ValueRange{outputRoot});
  Block *fusedBody = &fused.getBody().emplaceBlock();
  for (Value value : reads)
    fusedBody->addArgument(value.getType(), consumer.getLoc());
  fusedBody->addArgument(output.getType(), consumer.getLoc());

  auto fusedReadArg = [&](Value value) -> BlockArgument {
    for (auto indexed : llvm::enumerate(reads))
      if (indexed.value() == value)
        return fusedBody->getArgument(indexed.index());
    return BlockArgument();
  };
  IRMapping mapping;
  Block &producerBody = producer.getBody().front();
  Block &consumerBody = consumer.getBody().front();
  for (auto indexed : llvm::enumerate(producer.getWillReads()))
    mapping.map(producerBody.getArgument(indexed.index()),
                fusedReadArg(indexed.value()));
  mapping.map(producerBody.getArgument(producer.getWillReads().size()),
              fusedBody->getArgument(reads.size()));
  for (auto indexed : llvm::enumerate(consumer.getWillReads())) {
    BlockArgument sourceArg = consumerBody.getArgument(indexed.index());
    if (indexed.index() == pair.consumerInput)
      mapping.map(sourceArg, fusedBody->getArgument(reads.size()));
    else
      mapping.map(sourceArg, fusedReadArg(indexed.value()));
  }
  mapping.map(consumerBody.getArgument(consumer.getWillReads().size()),
              fusedBody->getArgument(reads.size()));

  TaskflowHyperblockOp producerHyperblock = findHyperblock(producer);
  TaskflowHyperblockOp consumerHyperblock = findHyperblock(consumer);
  if (!producerHyperblock || !consumerHyperblock ||
      producerHyperblock.getBody().empty() ||
      consumerHyperblock.getBody().empty())
    return reject(consumer, "fusion requires one hyperblock per task");
  Value forwarded;
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(fusedBody);
  for (Operation &operation : producerBody) {
    if (isa<TaskflowYieldOp>(&operation))
      continue;
    if (&operation == producerHyperblock.getOperation()) {
      SmallVector<Value> indices;
      for (Value index : producerHyperblock.getIndices())
        indices.push_back(mapping.lookupOrDefault(index));
      auto fusedHyperblock = bodyBuilder.create<TaskflowHyperblockOp>(
          consumer.getLoc(), TypeRange{}, indices, ValueRange{});
      Block *fusedHyperblockBody = &fusedHyperblock.getBody().emplaceBlock();
      Block &producerHyperblockBody = producerHyperblock.getBody().front();
      Block &consumerHyperblockBody = consumerHyperblock.getBody().front();
      for (BlockArgument argument : producerHyperblockBody.getArguments())
        fusedHyperblockBody->addArgument(argument.getType(), consumer.getLoc());
      for (auto indexed :
           llvm::enumerate(producerHyperblockBody.getArguments()))
        mapping.map(indexed.value(),
                    fusedHyperblockBody->getArgument(indexed.index()));
      for (auto indexed :
           llvm::enumerate(consumerHyperblockBody.getArguments()))
        if (indexed.index() < fusedHyperblockBody->getNumArguments())
          mapping.map(indexed.value(),
                      fusedHyperblockBody->getArgument(indexed.index()));

      OpBuilder hyperblockBuilder = OpBuilder::atBlockEnd(fusedHyperblockBody);
      BlockArgument producerOutput =
          producerBody.getArgument(producer.getWillReads().size());
      for (Operation &inner : producerHyperblockBody.without_terminator()) {
        if (auto store = dyn_cast<memref::StoreOp>(&inner)) {
          if (store.getMemRef() == producerOutput) {
            forwarded = mapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        hyperblockBuilder.clone(inner, mapping);
      }
      if (!forwarded)
        return reject(consumer,
                      "fusion requires one producer store to the intermediate");

      BlockArgument consumerIntermediate =
          consumerBody.getArgument(pair.consumerInput);
      for (Operation &inner : consumerHyperblockBody.without_terminator()) {
        if (auto load = dyn_cast<memref::LoadOp>(&inner)) {
          if (load.getMemRef() == consumerIntermediate) {
            mapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        hyperblockBuilder.clone(inner, mapping);
      }
      hyperblockBuilder.create<TaskflowHyperblockYieldOp>(consumer.getLoc());
      continue;
    }
    bodyBuilder.clone(operation, mapping);
  }
  addTaskYield(fusedBody, fusedBody->getArgument(reads.size()));

  copyCustomAttributes(producer, fused);
  markTask(
      builder, fused,
      pair.reduction ? "fused_reduction_consumer" : "fused_producer_consumer",
      pair.reduction ? "fused_reduction_consumer" : "fused_producer_consumer",
      producer->getAttrOfType<StringAttr>(kSemanticKPolicyAttr)
          ? producer->getAttrOfType<StringAttr>(kSemanticKPolicyAttr).getValue()
          : "sequential",
      producer->getAttrOfType<StringAttr>(kSemanticTopologyAttr)
          ? producer->getAttrOfType<StringAttr>(kSemanticTopologyAttr)
                .getValue()
          : "none");
  fused->setAttr(kSemanticFusionLayerAttr,
                 builder.getStringAttr("task_graph_fusion"));
  fused->setAttr(kSemanticFusionModeAttr,
                 builder.getStringAttr(pair.reduction ? "reduction_consumer"
                                                      : "sequential_last"));
  fused->setAttr(kSemanticFinalCOwnershipAttr,
                 builder.getStringAttr("external_final_D"));
  if (auto attr = producer->getAttr(kSemanticIncomingEdgesAttr))
    fused->setAttr(kSemanticIncomingEdgesAttr, attr);
  if (auto attr = producer->getAttr(kSemanticKRangeAttr))
    fused->setAttr(kSemanticKRangeAttr, attr);
  if (auto attr = producer->getAttr(kSemanticMRangeAttr))
    fused->setAttr(kSemanticMRangeAttr, attr);
  if (auto attr = producer->getAttr(kSemanticNRangeAttr))
    fused->setAttr(kSemanticNRangeAttr, attr);
  setRegionContracts(builder, fused, inputRegions, {outputRegion},
                     "fused_final");
  return fused;
}

// Clone a producer body while changing only the static K loop bounds and the
// memory state used as its initial accumulator.  The operation is deliberately
// restricted to the canonical body validated by analyzeSource.
static FailureOr<TaskflowTaskOp>
cloneSequentialProducer(OpBuilder &builder, const SourcePipeline &sourceInfo,
                        TaskflowTaskOp source, Value carry, Value outputRoot,
                        ValueRange roots, StringRef name, int64_t kLower,
                        int64_t kUpper, unsigned blockIndex,
                        unsigned blockCount, int64_t M, int64_t N, int64_t K) {
  SmallVector<Value> reads{source.getWillReads()[0], source.getWillReads()[1],
                           carry};
  auto task = builder.create<TaskflowTaskOp>(
      source.getLoc(), TypeRange{}, TypeRange{outputRoot.getType()},
      TypeRange{}, reads, ValueRange{outputRoot}, ValueRange{},
      builder.getStringAttr(name), roots, ValueRange{outputRoot});
  Block *body = &task.getBody().emplaceBlock();
  // original_*_memrefs are provenance operands, not task body parameters.
  for (Value operand : task.getWillReads())
    body->addArgument(operand.getType(), source.getLoc());
  for (Value operand : task.getWillWrites())
    body->addArgument(operand.getType(), source.getLoc());
  for (Value operand : task.getValueInputs())
    body->addArgument(operand.getType(), source.getLoc());

  IRMapping mapping;
  Block &sourceBody = source.getBody().front();
  for (auto indexed : llvm::enumerate(sourceBody.getArguments()))
    mapping.map(indexed.value(), body->getArgument(indexed.index()));
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
  Value newLower =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), kLower);
  Value newUpper =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), kUpper);
  Value newStep =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), 1);
  OpBuilder cloneBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : sourceBody)
    if (!isa<TaskflowYieldOp>(&operation))
      cloneBuilder.clone(operation, mapping);

  SmallVector<scf::ForOp> loops;
  task.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1)
    return reject(task, "cloned producer has no unique K loop");
  loops.front().getLowerBoundMutable().assign(newLower);
  loops.front().getUpperBoundMutable().assign(newUpper);
  loops.front().getStepMutable().assign(newStep);
  addTaskYield(body, body->getArgument(task.getWillReads().size()));

  copyCustomAttributes(source, task);
  markTask(builder, task, "gemm_producer", "producer", "sequential", "none");
  task->setAttr("amoeba.semantic.k_tiled", builder.getUnitAttr());
  task->setAttr("amoeba.semantic.k_block_index",
                builder.getI64IntegerAttr(blockIndex));
  task->setAttr("amoeba.semantic.k_block_count",
                builder.getI64IntegerAttr(blockCount));
  task->setAttr("amoeba.semantic.output_visibility",
                builder.getStringAttr("global_materialized"));
  task->setAttr(kSemanticInitialCOwnershipAttr,
                builder.getStringAttr(blockIndex == 0 ? "initial_C_once"
                                                      : "state_carry"));
  task->setAttr(kSemanticFinalCOwnershipAttr,
                builder.getStringAttr(blockIndex + 1 == blockCount
                                          ? "external_final_C"
                                          : "state_carry"));
  setRangeAttrs(builder, task, {0, M}, {0, N}, {kLower, kUpper});
  StaticRegion a = {{0, kLower}, {M, kUpper}};
  StaticRegion b = {{kLower, 0}, {kUpper, N}};
  setRegionContracts(builder, task, {a, b, fullRegion(sourceInfo)},
                     {fullRegion(sourceInfo)});
  return task;
}

} // namespace

namespace mlir::amoeba::neura {

template <typename Derived>
struct JointKPass : public PassWrapper<Derived, OperationPass<ModuleOp>> {
  JointKPass() = default;
  JointKPass(const JointKPass &other)
      : PassWrapper<Derived, OperationPass<ModuleOp>>(other) {}

  mlir::Pass::Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Taskflow function name"),
      llvm::cl::init("")};
  mlir::Pass::Option<int64_t> tileSize{
      *this, "tile-size",
      llvm::cl::desc("K block width; the final block may be smaller"),
      llvm::cl::init(0)};
  mlir::Pass::Option<int64_t> factor{
      *this, "factor",
      llvm::cl::desc("Number of K blocks (used when tile-size is absent)"),
      llvm::cl::init(0)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, TaskflowDialect>();
  }

  FailureOr<int64_t> blockWidth(int64_t K, bool allowExact) {
    int64_t width = tileSize;
    if (width == 0 && factor != 0) {
      if (factor < 1)
        return failure();
      width = (K + factor - 1) / factor;
    }
    if (width == 0)
      width = std::max<int64_t>(1, K / 2);
    if (width < 1 || width > K || (!allowExact && width == K))
      return failure();
    return width;
  }
};

static LogicalResult runSequentialK(ModuleOp module, StringRef functionName,
                                    int64_t requestedWidth) {
  FailureOr<SourcePipeline> source = analyzeSource(module, functionName);
  if (failed(source))
    return failure();
  int64_t width = requestedWidth;
  if (width < 1 || width >= source->K)
    return reject(source->producer,
                  "sequential K tiling requires a width in [1,K)");
  unsigned blockCount = static_cast<unsigned>((source->K + width - 1) / width);
  OpBuilder builder(source->producer);
  Value carry = source->C;
  SmallVector<TaskflowTaskOp> blocks;
  for (unsigned index = 0; index < blockCount; ++index) {
    int64_t lower = index * width;
    int64_t upper = std::min<int64_t>(source->K, lower + width);
    SmallVector<Value> roots{source->A, source->B,
                             index == 0 ? source->C : source->T};
    std::string name =
        (Twine(source->producer.getTaskName()) + ".k." + Twine(index)).str();
    FailureOr<TaskflowTaskOp> block = cloneSequentialProducer(
        builder, *source, source->producer, carry, source->T, roots, name,
        lower, upper, index, blockCount, source->M, source->N, source->K);
    if (failed(block))
      return failure();
    if (index != 0) {
      setIncomingEdges(
          builder, *block,
          {{blocks.back().getTaskName(), "state_carry", "tile_local"}});
    } else {
      setIncomingEdges(builder, *block, {});
    }
    blocks.push_back(*block);
    carry = block->getDoneWrites().front();
  }

  std::optional<unsigned> inputIndex =
      findInput(source->consumer, source->producer.getDoneWrites().front());
  if (!inputIndex)
    return reject(source->consumer, "cannot find the consumer T input");
  source->consumer.getWillReadsMutable().slice(*inputIndex, 1).assign(carry);
  markConsumerContract(builder, source->consumer, *source,
                       blocks.back().getTaskName(), "producer_consumer");
  setFunctionRewriteAttrs(builder, source->function, "sequential", "none");
  source->producer.erase();
  return success();
}

static LogicalResult runParallelK(ModuleOp module, StringRef functionName,
                                  int64_t requestedWidth, StringRef topology) {
  FailureOr<SourcePipeline> source = analyzeSource(module, functionName);
  if (failed(source))
    return failure();
  if (requestedWidth < 1 || requestedWidth > source->K)
    return reject(source->producer,
                  "parallel K tiling requires a width in [1,K]");
  unsigned blockCount =
      static_cast<unsigned>((source->K + requestedWidth - 1) / requestedWidth);
  OpBuilder builder(source->producer);
  for (unsigned index = 0; index < blockCount; ++index) {
    int64_t lower = index * requestedWidth;
    int64_t upper = std::min<int64_t>(source->K, lower + requestedWidth);
    auto alloc = builder.create<memref::AllocOp>(source->producer.getLoc(),
                                                 source->tensorType);
    alloc->setAttr(kSemanticPrivatePartialAttr, builder.getUnitAttr());
    alloc->setAttr("amoeba.semantic.logical_k_block",
                   builder.getI64IntegerAttr(index));
    std::string name = (Twine("partial_k.") + Twine(index)).str();
    FailureOr<TaskflowTaskOp> partial =
        buildPartialProducer(builder, *source, alloc.getMemref(), name, lower,
                             upper, index, topology);
    if (failed(partial))
      return failure();
  }

  // Keep the original producer as a temporary owner of T.  This preserves a
  // valid SSA state between standalone passes; the materialization pass below
  // consumes these private partial states, publishes the numeric reduction,
  // rewires the consumer, and then removes this pending producer.
  source->producer->setAttr(kSemanticKPolicyAttr,
                            builder.getStringAttr("parallel"));
  source->producer->setAttr(kSemanticTopologyAttr,
                            builder.getStringAttr(topology));
  source->producer->setAttr("amoeba.semantic.parallel_partials_pending",
                            builder.getUnitAttr());
  source->producer->setAttr("amoeba.semantic.k_block_count",
                            builder.getI64IntegerAttr(blockCount));
  setFunctionRewriteAttrs(builder, source->function, "parallel", topology);
  return success();
}

static FailureOr<SmallVector<TaskflowTaskOp>>
collectParallelPartialTasks(func::FuncOp function) {
  SmallVector<TaskflowTaskOp> partials;
  for (TaskflowTaskOp task : function.getOps<TaskflowTaskOp>()) {
    if (!task->hasAttr(kSemanticPrivatePartialAttr) ||
        taskKind(task) != "gemm_producer")
      continue;
    if (!task->getAttrOfType<IntegerAttr>("amoeba.semantic.logical_k_block"))
      return reject(task, "parallel partial is missing its K block index");
    partials.push_back(task);
  }
  if (partials.size() < 2)
    return reject(function,
                  "numeric reduction requires at least two private K partials");
  llvm::sort(partials, [](TaskflowTaskOp lhs, TaskflowTaskOp rhs) {
    return lhs->getAttrOfType<IntegerAttr>("amoeba.semantic.logical_k_block")
               .getInt() <
           rhs->getAttrOfType<IntegerAttr>("amoeba.semantic.logical_k_block")
               .getInt();
  });
  for (auto indexed : llvm::enumerate(partials)) {
    auto index = indexed.value()->getAttrOfType<IntegerAttr>(
        "amoeba.semantic.logical_k_block");
    if (index.getInt() != static_cast<int64_t>(indexed.index()))
      return reject(indexed.value(),
                    "parallel partial K blocks are not contiguous");
  }
  return partials;
}

static LogicalResult runMaterializeParallel(ModuleOp module,
                                            StringRef functionName) {
  FailureOr<SourcePipeline> source = analyzeSource(module, functionName);
  if (failed(source))
    return failure();
  auto pending = source->producer->getAttrOfType<UnitAttr>(
      "amoeba.semantic.parallel_partials_pending");
  if (!pending)
    return reject(source->producer, "numeric reduction materialization "
                                    "requires a pending parallel K tiling");
  auto topologyAttr =
      source->producer->getAttrOfType<StringAttr>(kSemanticTopologyAttr);
  if (!topologyAttr || (topologyAttr.getValue() != "linear" &&
                        topologyAttr.getValue() != "balanced_tree"))
    return reject(source->producer,
                  "pending parallel K tiling has no recognized topology");

  FailureOr<SmallVector<TaskflowTaskOp>> partials =
      collectParallelPartialTasks(source->function);
  if (failed(partials))
    return failure();
  SmallVector<Value> reducedStates;
  SmallVector<Value> reducedRoots;
  for (TaskflowTaskOp partial : *partials) {
    if (partial.getWillWrites().size() != 1 ||
        partial.getDoneWrites().size() != 1)
      return reject(partial, "parallel partial must have one write state");
    reducedStates.push_back(partial.getDoneWrites().front());
    reducedRoots.push_back(partial.getWillWrites().front());
  }

  OpBuilder builder(source->producer);
  if (topologyAttr.getValue() == "balanced_tree") {
    unsigned level = 0;
    while (reducedStates.size() > 1) {
      SmallVector<Value> nextStates;
      SmallVector<Value> nextRoots;
      for (unsigned index = 0; index < reducedStates.size(); index += 2) {
        if (index + 1 == reducedStates.size()) {
          nextStates.push_back(reducedStates[index]);
          nextRoots.push_back(reducedRoots[index]);
          continue;
        }
        auto alloc = builder.create<memref::AllocOp>(source->producer.getLoc(),
                                                     source->tensorType);
        alloc->setAttr(kSemanticPrivatePartialAttr, builder.getUnitAttr());
        std::string name =
            (Twine("reduction.tree.") + Twine(level) + "." + Twine(index / 2))
                .str();
        SmallVector<std::tuple<StringRef, StringRef, StringRef>> incoming;
        auto lhs = reducedStates[index].getDefiningOp<TaskflowTaskOp>();
        auto rhs = reducedStates[index + 1].getDefiningOp<TaskflowTaskOp>();
        if (!lhs || !rhs)
          return reject(source->producer,
                        "tree reduction input has no Taskflow producer");
        auto lhsKRange =
            lhs->getAttrOfType<DenseI64ArrayAttr>(kSemanticKRangeAttr);
        auto rhsKRange =
            rhs->getAttrOfType<DenseI64ArrayAttr>(kSemanticKRangeAttr);
        if (!lhsKRange || !rhsKRange || lhsKRange.asArrayRef().size() != 2 ||
            rhsKRange.asArrayRef().size() != 2 ||
            lhsKRange.asArrayRef()[1] != rhsKRange.asArrayRef()[0])
          return reject(source->producer,
                        "tree reduction inputs lack contiguous K ranges");
        incoming.push_back(
            {lhs.getTaskName(), "partial_reduction", "tile_local"});
        incoming.push_back(
            {rhs.getTaskName(), "partial_reduction", "tile_local"});
        FailureOr<TaskflowTaskOp> reduction = buildNumericReduction(
            builder, *source, {reducedStates[index], reducedStates[index + 1]},
            {reducedRoots[index], reducedRoots[index + 1]}, name,
            "balanced_tree", alloc.getMemref(), false, false, false, incoming);
        if (failed(reduction))
          return failure();
        setRangeAttrs(builder, *reduction, {0, source->M}, {0, source->N},
                      {lhsKRange.asArrayRef()[0], rhsKRange.asArrayRef()[1]});
        nextStates.push_back(reduction->getDoneWrites().front());
        nextRoots.push_back(alloc.getMemref());
      }
      reducedStates = std::move(nextStates);
      reducedRoots = std::move(nextRoots);
      ++level;
    }
  }

  SmallVector<std::tuple<StringRef, StringRef, StringRef>> incoming;
  for (Value state : reducedStates) {
    auto producer = state.getDefiningOp<TaskflowTaskOp>();
    if (!producer)
      return reject(source->producer,
                    "parallel reduction state has no Taskflow producer");
    incoming.push_back(
        {producer.getTaskName(), "partial_reduction", "tile_local"});
  }
  FailureOr<TaskflowTaskOp> finalReduction = buildNumericReduction(
      builder, *source, reducedStates, reducedRoots, "reduction.final",
      topologyAttr.getValue(), source->T, true, false, true, incoming);
  if (failed(finalReduction))
    return failure();

  std::optional<unsigned> inputIndex =
      findInput(source->consumer, source->producer.getDoneWrites().front());
  if (!inputIndex)
    return reject(source->consumer, "cannot find the consumer T input");
  source->consumer.getWillReadsMutable()
      .slice(*inputIndex, 1)
      .assign(finalReduction->getDoneWrites().front());
  markConsumerContract(builder, source->consumer, *source,
                       finalReduction->getTaskName(), "producer_consumer",
                       "parallel", topologyAttr.getValue());
  setFunctionRewriteAttrs(builder, source->function, "parallel",
                          topologyAttr.getValue());
  source->producer.erase();
  return success();
}

struct TileReductionKSequentialPass
    : public JointKPass<TileReductionKSequentialPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileReductionKSequentialPass)
  TileReductionKSequentialPass() = default;
  TileReductionKSequentialPass(const TileReductionKSequentialPass &other)
      : JointKPass<TileReductionKSequentialPass>(other) {}
  StringRef getArgument() const override {
    return "tile-reduction-k-sequential";
  }
  StringRef getDescription() const override {
    return "Split a static GEMM K reduction into a sequential Taskflow state "
           "chain";
  }
  void runOnOperation() override {
    FailureOr<SourcePipeline> source =
        analyzeSource(getOperation(), functionName);
    if (failed(source)) {
      signalPassFailure();
      return;
    }
    FailureOr<int64_t> width = blockWidth(source->K, false);
    if (failed(width) ||
        failed(runSequentialK(getOperation(), functionName, *width)))
      signalPassFailure();
  }
};

struct TileReductionKParallelLinearPass
    : public JointKPass<TileReductionKParallelLinearPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileReductionKParallelLinearPass)
  TileReductionKParallelLinearPass() = default;
  TileReductionKParallelLinearPass(
      const TileReductionKParallelLinearPass &other)
      : JointKPass<TileReductionKParallelLinearPass>(other) {}
  StringRef getArgument() const override {
    return "tile-reduction-k-parallel-linear";
  }
  StringRef getDescription() const override {
    return "Create private K partials and one linear numeric reduction task";
  }
  void runOnOperation() override {
    FailureOr<SourcePipeline> source =
        analyzeSource(getOperation(), functionName);
    if (failed(source)) {
      signalPassFailure();
      return;
    }
    FailureOr<int64_t> width = blockWidth(source->K, true);
    if (failed(width) ||
        failed(runParallelK(getOperation(), functionName, *width, "linear")))
      signalPassFailure();
  }
};

struct TileReductionKParallelTreePass
    : public JointKPass<TileReductionKParallelTreePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileReductionKParallelTreePass)
  TileReductionKParallelTreePass() = default;
  TileReductionKParallelTreePass(const TileReductionKParallelTreePass &other)
      : JointKPass<TileReductionKParallelTreePass>(other) {}
  StringRef getArgument() const override {
    return "tile-reduction-k-parallel-tree";
  }
  StringRef getDescription() const override {
    return "Create private K partials and a balanced tree numeric reduction";
  }
  void runOnOperation() override {
    FailureOr<SourcePipeline> source =
        analyzeSource(getOperation(), functionName);
    if (failed(source)) {
      signalPassFailure();
      return;
    }
    FailureOr<int64_t> width = blockWidth(source->K, true);
    if (failed(width) || failed(runParallelK(getOperation(), functionName,
                                             *width, "balanced_tree")))
      signalPassFailure();
  }
};

struct MaterializeNumericReductionPass
    : public JointKPass<MaterializeNumericReductionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeNumericReductionPass)
  MaterializeNumericReductionPass() = default;
  MaterializeNumericReductionPass(const MaterializeNumericReductionPass &other)
      : JointKPass<MaterializeNumericReductionPass>(other) {}
  StringRef getArgument() const override {
    return "materialize-numeric-reduction";
  }
  StringRef getDescription() const override {
    return "Validate and publish the explicit numeric K reduction state";
  }

  void runOnOperation() override {
    FailureOr<func::FuncOp> function =
        selectFunction(getOperation(), functionName);
    if (failed(function)) {
      signalPassFailure();
      return;
    }
    bool hasPendingParallel = false;
    for (TaskflowTaskOp task : (*function).getOps<TaskflowTaskOp>())
      hasPendingParallel |=
          task->hasAttr("amoeba.semantic.parallel_partials_pending");
    if (hasPendingParallel) {
      if (failed(runMaterializeParallel(getOperation(), functionName)))
        signalPassFailure();
      return;
    }
    SmallVector<TaskflowTaskOp> partials;
    SmallVector<TaskflowTaskOp> reductions;
    TaskflowTaskOp finalReduction;
    (*function).walk([&](TaskflowTaskOp task) {
      if (task->hasAttr(kSemanticPrivatePartialAttr))
        partials.push_back(task);
      if (taskKind(task) == "reduction" &&
          task->hasAttr(kSemanticNumericReductionAttr)) {
        reductions.push_back(task);
        if (task->hasAttr(kSemanticReductionFinalAttr))
          finalReduction = task;
      }
    });
    if (partials.empty() || reductions.empty() || !finalReduction)
      (*function).emitError()
          << "numeric reduction requires private partial tasks and one "
             "explicit final reduction task";
    if (partials.empty() || reductions.empty() || !finalReduction) {
      signalPassFailure();
      return;
    }

    // Recheck the graph contract before publishing the completion attribute.
    // Every private partial must be consumed by a numeric reduction, and the
    // final reduction must publish exactly one global T state.
    for (TaskflowTaskOp partial : partials) {
      if (partial.getDoneWrites().size() != 1 ||
          partial.getOriginalWriteMemrefs().size() != 1 ||
          partial.getDoneWrites().front().use_empty()) {
        partial.emitError()
            << "private partial has no numeric reduction consumer";
        signalPassFailure();
        return;
      }
      bool consumedByReduction = true;
      for (OpOperand &use : partial.getDoneWrites().front().getUses()) {
        auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
        consumedByReduction &= task && taskKind(task) == "reduction" &&
                               task->hasAttr(kSemanticNumericReductionAttr);
      }
      if (!consumedByReduction) {
        partial.emitError()
            << "private partial is consumed outside numeric reduction";
        signalPassFailure();
        return;
      }
    }
    if (finalReduction.getWillWrites().size() != 1 ||
        finalReduction.getOriginalWriteMemrefs().size() != 1 ||
        finalReduction.getDoneWrites().front().use_empty()) {
      finalReduction.emitError()
          << "final numeric reduction must publish a live T state";
      signalPassFailure();
      return;
    }
    OpBuilder builder(*function);
    finalReduction->setAttr("amoeba.semantic.numeric_reduction_materialized",
                            builder.getUnitAttr());
    (*function)->setAttr("amoeba.semantic.numeric_reduction_materialized",
                         builder.getUnitAttr());
  }
};

struct FuseReductionConsumerPass
    : public JointKPass<FuseReductionConsumerPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseReductionConsumerPass)
  FuseReductionConsumerPass() = default;
  FuseReductionConsumerPass(const FuseReductionConsumerPass &other)
      : JointKPass<FuseReductionConsumerPass>(other) {}
  StringRef getArgument() const override { return "fuse-reduction-consumer"; }
  StringRef getDescription() const override {
    return "Fuse only the completed K reduction with the elementwise consumer";
  }

  void runOnOperation() override {
    FailureOr<FusionPair> pair =
        findReductionFusionPair(getOperation(), functionName);
    if (failed(pair)) {
      signalPassFailure();
      return;
    }
    OpBuilder builder(pair->consumer);
    FailureOr<TaskflowTaskOp> fused = buildFusedFinalTask(builder, *pair);
    if (failed(fused)) {
      signalPassFailure();
      return;
    }
    func::FuncOp function =
        pair->consumer.getOperation()->getParentOfType<func::FuncOp>();
    Value oldOutput = pair->consumer.getDoneWrites().front();
    oldOutput.replaceAllUsesWith(fused->getDoneWrites().front());
    pair->consumer.erase();
    pair->producer.erase();
    setFunctionRewriteAttrs(builder, function,
                            pair->reduction ? "parallel" : "sequential",
                            pair->reduction ? "numeric" : "none");
  }
};

std::unique_ptr<Pass> createTileReductionKSequentialPass() {
  return std::make_unique<TileReductionKSequentialPass>();
}

std::unique_ptr<Pass> createTileReductionKParallelLinearPass() {
  return std::make_unique<TileReductionKParallelLinearPass>();
}

std::unique_ptr<Pass> createTileReductionKParallelTreePass() {
  return std::make_unique<TileReductionKParallelTreePass>();
}

std::unique_ptr<Pass> createMaterializeNumericReductionPass() {
  return std::make_unique<MaterializeNumericReductionPass>();
}

std::unique_ptr<Pass> createFuseReductionConsumerPass() {
  return std::make_unique<FuseReductionConsumerPass>();
}

} // namespace mlir::amoeba::neura
