//===- FissionTaskPass.cpp - Canonical hyperblock DFG fission ---*- C++ -*-===//
//
// Splits a hyperblock DFG at a stable predecessor-closed node partition and
// materializes every live scalar cut edge in a static tensor indexed by the
// unchanged counter domain.
// The two derived tasks therefore retain canonical counter/hyperblock form,
// and their temporary write/read state is an explicit Taskflow RAW edge.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kParentTaskAttr = "amoeba.fission.parent_task";
constexpr StringLiteral kLeftNodesAttr = "amoeba.fission.left_nodes";
constexpr StringLiteral kPartIndexAttr = "amoeba.fission.part_index";

struct FissionPlan {
  TaskflowTaskOp task;
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  TaskflowYieldOp taskYield;
  SmallVector<Operation *> nodes;
  SmallVector<Value> interfaceValues;
  SmallVector<int64_t> shape;
  SmallVector<unsigned> leftNodes;
  SmallVector<unsigned> rightNodes;
};

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

static bool isStructuralOrLineageAttribute(StringRef name) {
  return name == "task_name" || name == "operandSegmentSizes" ||
         name == "resultSegmentSizes" || name == "operand_segment_sizes" ||
         name == "result_segment_sizes" ||
         name.starts_with("amoeba.fission.") ||
         name.starts_with("amoeba.tiling.");
}

static bool isStorableScalar(Type type) { return type.isIntOrIndexOrFloat(); }

static bool hasDirectIndices(ValueRange indices, Block &hyperblockBody) {
  if (indices.size() != hyperblockBody.getNumArguments())
    return false;
  for (auto [index, value] : llvm::enumerate(indices))
    if (value != hyperblockBody.getArgument(index))
      return false;
  return true;
}

static FailureOr<FissionPlan> analyzeFission(TaskflowTaskOp task,
                                             ArrayRef<unsigned> leftNodes,
                                             std::string &error) {
  FissionPlan plan;
  plan.task = task;
  plan.leftNodes.append(leftNodes.begin(), leftNodes.end());
  if (!task || !task.getBody().hasOneBlock()) {
    error = "fission requires a selected single-block task";
    return failure();
  }
  if (!task.getDoneReads().empty() || task.getWillWrites().size() != 1 ||
      task.getDoneWrites().size() != 1 || !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != 1) {
    error = "fission requires one tensor write result and no read/value result";
    return failure();
  }
  for (NamedAttribute attribute : task->getAttrs())
    if (!isStructuralOrLineageAttribute(attribute.getName().strref())) {
      error = "fission rejects unsupported task state attribute " +
              attribute.getName().str();
      return failure();
    }

  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      plan.counters.push_back(counter);
    } else if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (plan.hyperblock) {
        error = "fission requires exactly one hyperblock";
        return failure();
      }
      plan.hyperblock = hyperblock;
    } else if (auto yield = dyn_cast<TaskflowYieldOp>(&operation)) {
      plan.taskYield = yield;
    } else if (!isa<arith::ConstantOp>(&operation)) {
      error = "fission requires canonical counters and one hyperblock";
      return failure();
    }
  }
  if (!plan.hyperblock || !plan.taskYield || plan.counters.empty() ||
      !plan.hyperblock.getIterArgs().empty() ||
      plan.hyperblock.getNumResults() != 0 ||
      plan.hyperblock.getIndices().size() != plan.counters.size()) {
    error = "fission rejects recurrence or a non-canonical counter domain";
    return failure();
  }
  Block &taskBody = task.getBody().front();
  size_t expectedArguments = task.getWillReads().size() +
                             task.getWillWrites().size() +
                             task.getValueInputs().size();
  if (taskBody.getNumArguments() != expectedArguments ||
      !plan.taskYield.getDoneReads().empty() ||
      plan.taskYield.getDoneWrites().size() != 1 ||
      plan.taskYield.getDoneWrites().front() !=
          taskBody.getArgument(task.getWillReads().size()) ||
      !plan.taskYield.getValueResults().empty()) {
    error = "fission requires task operands and yield to directly mirror "
            "tensor state";
    return failure();
  }
  for (auto [index, counter] : llvm::enumerate(plan.counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *lower != 0 || *step != 1 ||
        *upper <= *lower || (index == 0 && counter.getParentIndex()) ||
        (index != 0 && counter.getParentIndex() !=
                           plan.counters[index - 1].getCounterIndex()) ||
        plan.hyperblock.getIndices()[index] != counter.getCounterIndex()) {
      error = "fission requires a zero-based static unit-step counter chain";
      return failure();
    }
    plan.shape.push_back(*upper);
  }

  Block &hyperblockBody = plan.hyperblock.getBody().front();
  DenseSet<Value> arguments(hyperblockBody.getArguments().begin(),
                            hyperblockBody.getArguments().end());
  DenseMap<Value, unsigned> producers;
  for (Operation &operation : hyperblockBody.without_terminator()) {
    if (operation.getNumRegions() != 0 ||
        (!isa<memref::LoadOp, memref::StoreOp>(&operation) &&
         (!isPure(&operation) || !isMemoryEffectFree(&operation)))) {
      error = "fission rejects nested regions and unknown effects";
      return failure();
    }
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      bool knownRead = false;
      for (size_t index = 0; index < task.getWillReads().size(); ++index)
        knownRead |= load.getMemRef() == taskBody.getArgument(index);
      if (!knownRead || !hasDirectIndices(load.getIndices(), hyperblockBody)) {
        error = "fission requires direct counter-indexed tensor loads";
        return failure();
      }
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      if (store.getMemRef() !=
              taskBody.getArgument(task.getWillReads().size()) ||
          !hasDirectIndices(store.getIndices(), hyperblockBody)) {
        error = "fission requires direct counter-indexed tensor stores";
        return failure();
      }
    }
    unsigned nodeIndex = plan.nodes.size();
    for (Value operand : operation.getOperands()) {
      if (arguments.contains(operand) || isa<BlockArgument>(operand))
        continue;
      auto producer = producers.find(operand);
      if (producer == producers.end() || producer->second >= nodeIndex) {
        error = "fission rejects a backward or unknown DFG edge";
        return failure();
      }
    }
    for (Value result : operation.getResults())
      producers[result] = nodeIndex;
    plan.nodes.push_back(&operation);
  }
  if (leftNodes.empty() || leftNodes.size() >= plan.nodes.size()) {
    error =
        "left-nodes must select a non-empty proper hyperblock DFG partition";
    return failure();
  }
  SmallVector<bool> isLeft(plan.nodes.size(), false);
  unsigned previous = 0;
  for (auto [position, index] : llvm::enumerate(leftNodes)) {
    if (index >= plan.nodes.size() || (position != 0 && index <= previous)) {
      error = "left-nodes must be unique increasing hyperblock node ordinals";
      return failure();
    }
    previous = index;
    isLeft[index] = true;
  }
  for (unsigned index = 0; index < plan.nodes.size(); ++index) {
    if (!isLeft[index]) {
      plan.rightNodes.push_back(index);
      continue;
    }
    if (isa<memref::StoreOp>(plan.nodes[index])) {
      error = "fission rejects a cut after an externally visible memory write";
      return failure();
    }
    for (Value operand : plan.nodes[index]->getOperands()) {
      auto producer = producers.find(operand);
      if (producer != producers.end() && !isLeft[producer->second]) {
        error = "left-nodes must be predecessor closed";
        return failure();
      }
    }
  }
  if (!llvm::any_of(plan.rightNodes, [&](unsigned index) {
        return isa<memref::StoreOp>(plan.nodes[index]);
      })) {
    error = "fission suffix must retain the tensor write";
    return failure();
  }

  DenseSet<Value> interfaceSet;
  for (unsigned index : plan.rightNodes)
    for (Value operand : plan.nodes[index]->getOperands()) {
      auto producer = producers.find(operand);
      if (producer != producers.end() && isLeft[producer->second]) {
        if (!isStorableScalar(operand.getType())) {
          error = "fission cut edge has a non-storable value type";
          return failure();
        }
        interfaceSet.insert(operand);
      }
    }
  for (unsigned index : plan.leftNodes)
    for (Value result : plan.nodes[index]->getResults())
      if (interfaceSet.contains(result))
        plan.interfaceValues.push_back(result);
  if (plan.interfaceValues.empty()) {
    error = "fission cut has no live DFG edge";
    return failure();
  }
  return plan;
}

static void setLineage(TaskflowTaskOp source, TaskflowTaskOp target,
                       OpBuilder &builder, ArrayRef<unsigned> leftNodes,
                       unsigned partIndex) {
  target->setAttr(kParentTaskAttr, builder.getStringAttr(source.getTaskName()));
  SmallVector<int64_t> ordinals(leftNodes.begin(), leftNodes.end());
  target->setAttr(kLeftNodesAttr, builder.getDenseI64ArrayAttr(ordinals));
  target->setAttr(kPartIndexAttr, builder.getI64IntegerAttr(partIndex));
}

static void cloneScaffold(FissionPlan &plan, TaskflowTaskOp target,
                          IRMapping &mapping,
                          SmallVectorImpl<TaskflowCounterOp> &counters,
                          OpBuilder &builder) {
  Block &sourceBody = plan.task.getBody().front();
  for (Operation &operation : sourceBody) {
    if (isa<TaskflowHyperblockOp, TaskflowYieldOp>(&operation))
      continue;
    Operation *clone = builder.clone(operation, mapping);
    if (auto counter = dyn_cast<TaskflowCounterOp>(clone))
      counters.push_back(counter);
  }
}

static TaskflowHyperblockOp
createHyperblock(FissionPlan &plan, ArrayRef<TaskflowCounterOp> counters,
                 IRMapping &mapping, OpBuilder &builder) {
  SmallVector<Value> indices;
  llvm::transform(
      counters, std::back_inserter(indices),
      [](TaskflowCounterOp counter) { return counter.getCounterIndex(); });
  auto hyperblock = builder.create<TaskflowHyperblockOp>(
      plan.task.getLoc(), TypeRange{}, indices, ValueRange{});
  Block *body = &hyperblock.getBody().emplaceBlock();
  for (BlockArgument sourceArgument :
       plan.hyperblock.getBody().front().getArguments()) {
    BlockArgument argument =
        body->addArgument(sourceArgument.getType(), plan.task.getLoc());
    mapping.map(sourceArgument, argument);
  }
  return hyperblock;
}

static TaskflowTaskOp createPrefix(FissionPlan &plan, OpBuilder &builder,
                                   ValueRange temporaries) {
  SmallVector<Type> resultTypes(temporaries.getTypes());
  SmallVector<Value> originalWrites(temporaries);
  auto prefix = builder.create<TaskflowTaskOp>(
      plan.task.getLoc(), TypeRange{}, resultTypes, TypeRange{},
      plan.task.getWillReads(), temporaries, plan.task.getValueInputs(),
      builder.getStringAttr(
          (Twine(plan.task.getTaskName()) + ".split.0").str()),
      plan.task.getOriginalReadMemrefs(), originalWrites);
  Block *body = new Block();
  prefix.getBody().push_back(body);
  IRMapping mapping;
  SmallVector<Value> operands(plan.task.getWillReads());
  llvm::append_range(operands, temporaries);
  llvm::append_range(operands, plan.task.getValueInputs());
  for (Value operand : operands)
    body->addArgument(operand.getType(), plan.task.getLoc());
  Block &sourceBody = plan.task.getBody().front();
  for (size_t index = 0; index < plan.task.getWillReads().size(); ++index)
    mapping.map(sourceBody.getArgument(index), body->getArgument(index));
  size_t sourceValueOffset =
      plan.task.getWillReads().size() + plan.task.getWillWrites().size();
  size_t targetValueOffset =
      plan.task.getWillReads().size() + temporaries.size();
  for (size_t index = 0; index < plan.task.getValueInputs().size(); ++index)
    mapping.map(sourceBody.getArgument(sourceValueOffset + index),
                body->getArgument(targetValueOffset + index));

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<TaskflowCounterOp> counters;
  cloneScaffold(plan, prefix, mapping, counters, bodyBuilder);
  auto hyperblock = createHyperblock(plan, counters, mapping, bodyBuilder);
  Block &hyperblockBody = hyperblock.getBody().front();
  OpBuilder hyperblockBuilder = OpBuilder::atBlockEnd(&hyperblockBody);
  for (unsigned index : plan.leftNodes)
    hyperblockBuilder.clone(*plan.nodes[index], mapping);
  for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues)) {
    Value temporary =
        body->getArgument(plan.task.getWillReads().size() + index);
    hyperblockBuilder.create<memref::StoreOp>(
        plan.task.getLoc(), mapping.lookup(interfaceValue), temporary,
        hyperblockBody.getArguments());
  }
  hyperblockBuilder.create<TaskflowHyperblockYieldOp>(plan.task.getLoc());
  bodyBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> yields;
  for (size_t index = 0; index < temporaries.size(); ++index)
    yields.push_back(
        body->getArgument(plan.task.getWillReads().size() + index));
  bodyBuilder.create<TaskflowYieldOp>(plan.task.getLoc(), ValueRange{}, yields,
                                      ValueRange{});
  setLineage(plan.task, prefix, builder, plan.leftNodes, 0);
  return prefix;
}

static TaskflowTaskOp createSuffix(FissionPlan &plan, OpBuilder &builder,
                                   TaskflowTaskOp prefix) {
  SmallVector<Value> reads(prefix.getDoneWrites());
  llvm::append_range(reads, plan.task.getWillReads());
  SmallVector<Value> originalReads(prefix.getWillWrites());
  llvm::append_range(originalReads, plan.task.getOriginalReadMemrefs());
  auto suffix = builder.create<TaskflowTaskOp>(
      plan.task.getLoc(), TypeRange{}, plan.task.getDoneWrites().getTypes(),
      TypeRange{}, reads, plan.task.getWillWrites(), plan.task.getValueInputs(),
      builder.getStringAttr(
          (Twine(plan.task.getTaskName()) + ".split.1").str()),
      originalReads, plan.task.getOriginalWriteMemrefs());
  Block *body = new Block();
  suffix.getBody().push_back(body);
  for (Value operand : suffix->getOperands()) {
    if (body->getNumArguments() == reads.size() +
                                       plan.task.getWillWrites().size() +
                                       plan.task.getValueInputs().size())
      break;
    body->addArgument(operand.getType(), plan.task.getLoc());
  }
  IRMapping mapping;
  Block &sourceBody = plan.task.getBody().front();
  size_t temporaryCount = prefix.getDoneWrites().size();
  for (size_t index = 0; index < plan.task.getWillReads().size(); ++index)
    mapping.map(sourceBody.getArgument(index),
                body->getArgument(temporaryCount + index));
  size_t sourceWriteOffset = plan.task.getWillReads().size();
  size_t targetWriteOffset = reads.size();
  mapping.map(sourceBody.getArgument(sourceWriteOffset),
              body->getArgument(targetWriteOffset));
  size_t sourceValueOffset =
      sourceWriteOffset + plan.task.getWillWrites().size();
  size_t targetValueOffset =
      targetWriteOffset + plan.task.getWillWrites().size();
  for (size_t index = 0; index < plan.task.getValueInputs().size(); ++index)
    mapping.map(sourceBody.getArgument(sourceValueOffset + index),
                body->getArgument(targetValueOffset + index));

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<TaskflowCounterOp> counters;
  cloneScaffold(plan, suffix, mapping, counters, bodyBuilder);
  auto hyperblock = createHyperblock(plan, counters, mapping, bodyBuilder);
  Block &hyperblockBody = hyperblock.getBody().front();
  OpBuilder hyperblockBuilder = OpBuilder::atBlockEnd(&hyperblockBody);
  for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues)) {
    Value loaded = hyperblockBuilder.create<memref::LoadOp>(
        plan.task.getLoc(), body->getArgument(index),
        hyperblockBody.getArguments());
    mapping.map(interfaceValue, loaded);
  }
  for (unsigned index : plan.rightNodes)
    hyperblockBuilder.clone(*plan.nodes[index], mapping);
  hyperblockBuilder.create<TaskflowHyperblockYieldOp>(plan.task.getLoc());
  bodyBuilder.setInsertionPointToEnd(body);
  Value output = body->getArgument(reads.size());
  bodyBuilder.create<TaskflowYieldOp>(plan.task.getLoc(), ValueRange{},
                                      ValueRange{output}, ValueRange{});
  setLineage(plan.task, suffix, builder, plan.leftNodes, 1);
  return suffix;
}

static LogicalResult rewriteFission(FissionPlan &plan, func::FuncOp function) {
  std::string prefixName = (Twine(plan.task.getTaskName()) + ".split.0").str();
  std::string suffixName = (Twine(plan.task.getTaskName()) + ".split.1").str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    collision |= task != plan.task && (task.getTaskName() == prefixName ||
                                       task.getTaskName() == suffixName);
  });
  if (collision) {
    plan.task.emitError(
        "fission-derived task name collides with an existing task");
    return failure();
  }

  OpBuilder builder(plan.task);
  SmallVector<Value> temporaries;
  for (Value interfaceValue : plan.interfaceValues) {
    auto type = MemRefType::get(plan.shape, interfaceValue.getType());
    temporaries.push_back(
        builder.create<memref::AllocOp>(plan.task.getLoc(), type));
  }
  TaskflowTaskOp prefix = createPrefix(plan, builder, temporaries);
  TaskflowTaskOp suffix = createSuffix(plan, builder, prefix);
  for (auto [oldResult, newResult] :
       llvm::zip(plan.task.getDoneWrites(), suffix.getDoneWrites()))
    oldResult.replaceAllUsesWith(newResult);
  plan.task.erase();
  return success();
}

struct FissionTaskPass
    : public PassWrapper<FissionTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FissionTaskPass)
  FissionTaskPass() = default;
  FissionTaskPass(const FissionTaskPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "fission-task"; }
  StringRef getDescription() const override {
    return "Split a canonical hyperblock DFG through static cut tensors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  Option<std::string> taskName{*this, "task-name",
                               llvm::cl::desc("Exact Taskflow task name"),
                               llvm::cl::init("")};
  Option<std::string> leftNodes{
      *this, "left-nodes",
      llvm::cl::desc("Colon-separated predecessor-closed DFG node ordinals"),
      llvm::cl::init("")};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<TaskflowTaskOp> matches;
    function.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == taskName)
        matches.push_back(task);
    });
    SmallVector<unsigned> parsedLeftNodes;
    std::stringstream stream(leftNodes.getValue());
    std::string token;
    while (std::getline(stream, token, ':')) {
      if (token.empty() || !llvm::all_of(token, llvm::isDigit)) {
        function.emitError("fission-task left-nodes must be colon-separated "
                           "non-negative integers");
        return signalPassFailure();
      }
      uint64_t ordinal = 0;
      if (StringRef(token).getAsInteger(10, ordinal) ||
          ordinal > std::numeric_limits<unsigned>::max()) {
        function.emitError("fission-task left-nodes ordinal is out of range");
        return signalPassFailure();
      }
      parsedLeftNodes.push_back(static_cast<unsigned>(ordinal));
    }
    if (taskName.empty() || parsedLeftNodes.empty() || matches.size() != 1) {
      function.emitError()
          << "fission-task requires one task-name and non-empty left-nodes";
      return signalPassFailure();
    }
    std::string error;
    FailureOr<FissionPlan> plan =
        analyzeFission(matches.front(), parsedLeftNodes, error);
    if (failed(plan)) {
      matches.front().emitError() << error;
      return signalPassFailure();
    }
    if (failed(rewriteFission(*plan, function)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createFissionTaskPass() {
  return std::make_unique<FissionTaskPass>();
}
} // namespace mlir::amoeba::neura
