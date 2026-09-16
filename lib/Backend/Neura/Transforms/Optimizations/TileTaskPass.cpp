//===- TileTaskPass.cpp - Static counter-domain task tiling -----*- C++ -*-===//
//
// Splits a canonical pre-lowering Taskflow task into exactly k tasks along
// one counter axis. Each part retains the counter chain and hyperblock. The
// selected intervals are disjoint and a taskflow.join combines their states.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kParentTaskAttr = "amoeba.tiling.parent_task";
constexpr StringLiteral kAxisAttr = "amoeba.tiling.axis";
constexpr StringLiteral kFactorAttr = "amoeba.tiling.factor";
constexpr StringLiteral kPartIndexAttr = "amoeba.tiling.part_index";
constexpr StringLiteral kOriginalRangeAttr = "amoeba.tiling.original_range";
constexpr StringLiteral kDerivedRangeAttr = "amoeba.tiling.derived_range";

struct TilePlan {
  TaskflowTaskOp task;
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  int64_t axis = 0;
  int64_t factor = 0;
  int64_t lower = 0;
  int64_t upper = 0;
};

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError(message);
  return failure();
}

static bool isStructuralOrLineageAttribute(StringRef name) {
  return name == "operandSegmentSizes" || name == "resultSegmentSizes" ||
         name == "operand_segment_sizes" || name == "result_segment_sizes" ||
         name == "task_name" || name.starts_with("amoeba.tiling.") ||
         name.starts_with("amoeba.fission.");
}

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
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

static FailureOr<TilePlan> analyzeTask(TaskflowTaskOp task, int64_t axis,
                                       int64_t factor) {
  if (!task || !task.getBody().hasOneBlock())
    return failure();
  for (NamedAttribute attribute : task->getAttrs())
    if (!isStructuralOrLineageAttribute(attribute.getName().strref()))
      return reject(task, Twine("tiling rejects unsupported task state '") +
                              attribute.getName().getValue() + "'");
  if (task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      !task.getDoneReads().empty() || !task.getValueOutputs().empty() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return reject(
        task,
        "tiling requires one output state and no yielded read/value state");

  TilePlan plan;
  plan.task = task;
  plan.axis = axis;
  plan.factor = factor;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      plan.counters.push_back(counter);
      continue;
    }
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (plan.hyperblock)
        return reject(task, "tiling requires exactly one hyperblock");
      plan.hyperblock = hyperblock;
      continue;
    }
    if (isa<TaskflowYieldOp, arith::ConstantOp>(&operation))
      continue;
    return reject(task,
                  "tiling requires canonical counters and one hyperblock");
  }
  if (!plan.hyperblock || plan.counters.empty() || axis < 0 ||
      axis >= static_cast<int64_t>(plan.counters.size()) || factor < 2)
    return reject(
        task, "tiling requires a valid static counter axis and factor >= 2");

  for (auto [index, counter] : llvm::enumerate(plan.counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
      return reject(counter, "tiling requires constant unit-step counters");
    if ((index == 0 && counter.getParentIndex()) ||
        (index != 0 && counter.getParentIndex() !=
                           plan.counters[index - 1].getCounterIndex()))
      return reject(counter, "tiling requires one ordered counter chain");
  }
  if (plan.hyperblock.getIndices().size() != plan.counters.size())
    return reject(plan.hyperblock,
                  "hyperblock indices must match the counter chain");
  for (auto [index, value] : llvm::enumerate(plan.hyperblock.getIndices()))
    if (value != plan.counters[index].getCounterIndex())
      return reject(plan.hyperblock,
                    "hyperblock indices must match the counter chain");
  if (!plan.hyperblock.getIterArgs().empty() ||
      plan.hyperblock.getNumResults() != 0)
    return reject(plan.hyperblock, "tiling rejects loop-carried recurrence");

  SmallVector<Value> states(task.getWillReads());
  states.push_back(task.getWillWrites().front());
  for (Value state : states) {
    auto type = dyn_cast<MemRefType>(state.getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        type.getRank() != static_cast<int64_t>(plan.counters.size()))
      return reject(
          task,
          "tiling requires static identity memrefs matching counter rank");
    for (auto [counterAxis, counter] : llvm::enumerate(plan.counters)) {
      int64_t lower = *constantIndex(counter.getLowerBound());
      int64_t upper = *constantIndex(counter.getUpperBound());
      if (lower < 0 || upper <= lower || upper > type.getShape()[counterAxis])
        return reject(
            task, "counter domain must select a non-empty tensor subdomain");
    }
  }
  SmallVector<Value> roots(task.getOriginalReadMemrefs());
  roots.push_back(task.getOriginalWriteMemrefs().front());
  for (auto [state, root] : llvm::zip(states, roots))
    if (state.getType() != root.getType())
      return reject(task, "tiling state and original storage types differ");
  for (size_t first = 0; first < roots.size(); ++first)
    for (size_t second = first + 1; second < roots.size(); ++second)
      if (!provesDistinctStorage(roots[first], roots[second]))
        return reject(task, "tiling requires distinct allocations or "
                            "amoeba.noalias function arguments");

  TaskflowCounterOp selected = plan.counters[axis];
  plan.lower = *constantIndex(selected.getLowerBound());
  plan.upper = *constantIndex(selected.getUpperBound());
  if (factor > plan.upper - plan.lower)
    return reject(task, "tiling factor must not exceed the selected extent");
  return plan;
}

static TaskflowTaskOp createPart(OpBuilder &builder, TilePlan &plan,
                                 int64_t partIndex, int64_t partLower,
                                 int64_t partUpper) {
  TaskflowTaskOp source = plan.task;
  std::string name = (Twine(source.getTaskName()) + ".tile." +
                      Twine(plan.axis) + "." + Twine(partIndex))
                         .str();
  TaskflowTaskOp part = builder.create<TaskflowTaskOp>(
      source.getLoc(), TypeRange{}, source.getDoneWrites().getTypes(),
      TypeRange{}, source.getWillReads(), source.getWillWrites(),
      source.getValueInputs(), builder.getStringAttr(name),
      source.getOriginalReadMemrefs(), source.getOriginalWriteMemrefs());
  Block *body = new Block();
  part.getBody().push_back(body);
  IRMapping mapping;
  Block &sourceBody = source.getBody().front();
  for (auto [index, operand] : llvm::enumerate(source->getOperands())) {
    if (index >= sourceBody.getNumArguments())
      break;
    BlockArgument argument =
        body->addArgument(operand.getType(), source.getLoc());
    mapping.map(sourceBody.getArgument(index), argument);
  }
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : sourceBody) {
    if (!isa<TaskflowYieldOp>(&operation))
      bodyBuilder.clone(operation, mapping);
  }
  SmallVector<TaskflowCounterOp> clonedCounters;
  for (Operation &operation : *body)
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      clonedCounters.push_back(counter);
  if (clonedCounters.size() != plan.counters.size())
    llvm_unreachable("validated counter chain changed while cloning");
  bodyBuilder.setInsertionPoint(clonedCounters[plan.axis]);
  Value lower =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), partLower);
  Value upper =
      bodyBuilder.create<arith::ConstantIndexOp>(source.getLoc(), partUpper);
  clonedCounters[plan.axis].getLowerBoundMutable().assign(lower);
  clonedCounters[plan.axis].getUpperBoundMutable().assign(upper);
  bodyBuilder.setInsertionPointToEnd(body);
  bodyBuilder.create<TaskflowYieldOp>(
      source.getLoc(), ValueRange{},
      ValueRange{body->getArgument(source.getWillReads().size())},
      ValueRange{});
  part->setAttr(kParentTaskAttr, builder.getStringAttr(source.getTaskName()));
  part->setAttr(kAxisAttr, builder.getI64IntegerAttr(plan.axis));
  part->setAttr(kFactorAttr, builder.getI64IntegerAttr(plan.factor));
  part->setAttr(kPartIndexAttr, builder.getI64IntegerAttr(partIndex));
  part->setAttr(kOriginalRangeAttr,
                builder.getDenseI64ArrayAttr({plan.lower, plan.upper}));
  part->setAttr(kDerivedRangeAttr,
                builder.getDenseI64ArrayAttr({partLower, partUpper}));
  return part;
}

static LogicalResult rewriteTask(TilePlan &plan, func::FuncOp function) {
  for (int64_t part = 0; part < plan.factor; ++part) {
    std::string name = (Twine(plan.task.getTaskName()) + ".tile." +
                        Twine(plan.axis) + "." + Twine(part))
                           .str();
    bool collision = false;
    function.walk([&](TaskflowTaskOp task) {
      collision |= task != plan.task && task.getTaskName() == name;
    });
    if (collision)
      return reject(plan.task,
                    "tiling-derived task name collides with an existing task");
  }
  OpBuilder builder(plan.task);
  SmallVector<Value> tileStates;
  int64_t extent = plan.upper - plan.lower;
  int64_t base = extent / plan.factor;
  int64_t remainder = extent % plan.factor;
  int64_t cursor = plan.lower;
  for (int64_t part = 0; part < plan.factor; ++part) {
    int64_t width = base + (part < remainder ? 1 : 0);
    TaskflowTaskOp tile =
        createPart(builder, plan, part, cursor, cursor + width);
    tileStates.push_back(tile.getDoneWrites().front());
    cursor += width;
  }
  SmallVector<int64_t> regionLower;
  SmallVector<int64_t> regionUpper;
  for (TaskflowCounterOp counter : plan.counters) {
    regionLower.push_back(*constantIndex(counter.getLowerBound()));
    regionUpper.push_back(*constantIndex(counter.getUpperBound()));
  }
  auto join = builder.create<TaskflowJoinOp>(
      plan.task.getLoc(), plan.task.getDoneWrites().front().getType(),
      tileStates, plan.task.getWillWrites().front(),
      builder.getI64IntegerAttr(plan.axis),
      builder.getDenseI64ArrayAttr(regionLower),
      builder.getDenseI64ArrayAttr(regionUpper));
  plan.task.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  plan.task.erase();
  return success();
}

struct TileTaskPass
    : public PassWrapper<TileTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileTaskPass)
  TileTaskPass() = default;
  TileTaskPass(const TileTaskPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "tile-task"; }
  StringRef getDescription() const override {
    return "Partition a static Taskflow counter domain into disjoint tasks";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, TaskflowDialect>();
  }
  Option<std::string> taskName{*this, "task-name",
                               llvm::cl::desc("Exact Taskflow task name"),
                               llvm::cl::init("")};
  Option<int64_t> axis{*this, "axis", llvm::cl::desc("Counter axis"),
                       llvm::cl::init(-1)};
  Option<int64_t> factor{*this, "factor",
                         llvm::cl::desc("Number of disjoint domain parts"),
                         llvm::cl::init(0)};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<TaskflowTaskOp> matches;
    function.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == taskName)
        matches.push_back(task);
    });
    if (taskName.empty() || matches.size() != 1) {
      function.emitError()
          << "tile task name must select exactly one task (found "
          << matches.size() << ")";
      return signalPassFailure();
    }
    FailureOr<TilePlan> plan = analyzeTask(matches.front(), axis, factor);
    if (failed(plan))
      return signalPassFailure();
    if (failed(rewriteTask(*plan, function)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createTileTaskPass() {
  return std::make_unique<TileTaskPass>();
}
} // namespace mlir::amoeba::neura
