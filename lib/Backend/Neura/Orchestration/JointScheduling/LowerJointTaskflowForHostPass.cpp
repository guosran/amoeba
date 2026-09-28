//===- LowerJointTaskflowForHostPass.cpp ----------------------*- C++ -*-===//
//
// Lowers the executable subset of ORBIT Taskflow IR to ordinary host IR.
// Taskflow tasks are dependency-ordered and inlined, counters/hyperblocks are
// represented by nested scf.for operations, and proven completion joins are
// lowered to aliases of their base memref.  The pass is deliberately
// fail-closed: a body contract or an operation shape that cannot be lowered is
// diagnosed before any rewrite is made.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kErrorPrefix = "[ORBIT_HOST_LOWERING] ";
constexpr StringLiteral kHostTaskNameAttr = "amoeba.host.task_name";
constexpr StringLiteral kHostTaskCompletionAttr = "amoeba.host.task_completion";
constexpr StringLiteral kHostTraceSymbolAttr = "amoeba.host.trace_symbol";

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError() << kErrorPrefix << message;
  return failure();
}

static bool isTaskflowOperation(Operation *operation) {
  Dialect *dialect = operation->getDialect();
  return dialect && dialect->getNamespace() == "taskflow";
}

// The generated ORBIT bodies use arithmetic, memref access, and structured
// control flow.  Checking the dialect namespace keeps this whitelist useful as
// new arith/memref operations are introduced, while regions are still walked
// recursively below so an unrelated operation cannot hide inside a loop.
static bool isSupportedHostOperation(Operation *operation) {
  if (isa<scf::ForOp, scf::YieldOp>(operation))
    return true;
  Dialect *dialect = operation->getDialect();
  if (!dialect)
    return false;
  StringRef name = dialect->getNamespace();
  return name == "arith" || name == "memref";
}

static LogicalResult validateHostRegion(Region &region, Operation *owner,
                                        bool allowHyperblockYield = false) {
  if (!region.hasOneBlock())
    return reject(owner, "HOST_LOWERING_UNSUPPORTED_SHAPE: expected one block");

  Block &block = region.front();
  for (Operation &operation : block) {
    if (allowHyperblockYield && isa<TaskflowHyperblockYieldOp>(&operation) &&
        &operation == block.getTerminator())
      continue;
    if (!isSupportedHostOperation(&operation))
      return reject(&operation, "HOST_LOWERING_UNSUPPORTED_OPERATION: " +
                                    operation.getName().getStringRef());
    for (Region &nested : operation.getRegions())
      if (failed(validateHostRegion(nested, &operation)))
        return failure();
  }
  return success();
}

static LogicalResult validateCounter(TaskflowCounterOp counter) {
  if (!counter.getCounterIndex().getType().isIndex() ||
      !counter.getLowerBound().getType().isIndex() ||
      !counter.getUpperBound().getType().isIndex() ||
      !counter.getStep().getType().isIndex())
    return reject(counter,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: counter bounds and "
                  "result must have index type");
  return success();
}

static LogicalResult validateHyperblock(TaskflowHyperblockOp hyperblock,
                                        ArrayRef<TaskflowCounterOp> counters) {
  if (hyperblock.getIndices().empty() || !hyperblock.getIterArgs().empty() ||
      !hyperblock.getOutputs().empty())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock requires "
                  "counter indices and no iter_args/results");
  if (!hyperblock.getBody().hasOneBlock())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock body must "
                  "have one block");
  Block &body = hyperblock.getBody().front();
  if (body.getNumArguments() != hyperblock.getIndices().size())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock index/body "
                  "argument count mismatch");
  if (!isa<TaskflowHyperblockYieldOp>(body.getTerminator()))
    return reject(hyperblock,
                  "HOST_LOWERING_BODY_CONTRACT: hyperblock must end in "
                  "taskflow.hyperblock.yield");
  auto yield = cast<TaskflowHyperblockYieldOp>(body.getTerminator());
  if (!yield.getIterArgsNext().empty() || !yield.getResults().empty())
    return reject(yield, "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock yield "
                         "cannot carry values");

  llvm::DenseSet<Value> usedCounters;
  for (auto [index, blockArgument] :
       llvm::zip(hyperblock.getIndices(), body.getArguments())) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    if (!counter || !llvm::is_contained(counters, counter))
      return reject(hyperblock,
                    "HOST_LOWERING_BODY_CONTRACT: hyperblock index is not a "
                    "taskflow.counter result");
    if (!index.getType().isIndex() || !blockArgument.getType().isIndex())
      return reject(hyperblock,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock indices "
                    "must have index type");
    usedCounters.insert(index);

    for (OpOperand &use : index.getUses()) {
      Operation *user = use.getOwner();
      if (user == hyperblock.getOperation())
        continue;
      if (auto child = dyn_cast<TaskflowCounterOp>(user)) {
        if (child.getParentIndex() == index &&
            llvm::is_contained(counters, child))
          continue;
      }
      if (user->getParentOfType<TaskflowHyperblockOp>() != hyperblock)
        return reject(user,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: counter escapes its "
                      "hyperblock");
    }
  }
  if (usedCounters.size() != counters.size())
    return reject(hyperblock,
                  "HOST_LOWERING_BODY_CONTRACT: every counter must drive the "
                  "hyperblock exactly once");
  for (auto [position, index] : llvm::enumerate(hyperblock.getIndices())) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    Value parent = counter.getParentIndex();
    if ((position == 0 && parent) ||
        (position != 0 && parent != hyperblock.getIndices()[position - 1]))
      return reject(counter, "HOST_LOWERING_UNSUPPORTED_SHAPE: counter "
                             "parent must follow hyperblock nesting order");
  }
  if (failed(validateHostRegion(hyperblock.getBody(), hyperblock,
                                /*allowHyperblockYield=*/true)))
    return failure();
  return success();
}

static LogicalResult validateTask(TaskflowTaskOp task) {
  if (!task.getBody().hasOneBlock())
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: task body must have one "
                        "block");
  Block &body = task.getBody().front();
  SmallVector<Value> inputs;
  llvm::append_range(inputs, task.getWillReads());
  llvm::append_range(inputs, task.getWillWrites());
  llvm::append_range(inputs, task.getValueInputs());
  if (body.getNumArguments() != inputs.size())
    return reject(task,
                  "HOST_LOWERING_BODY_CONTRACT: body arguments must match "
                  "will_reads, will_writes, and value_inputs; "
                  "original_* operands are not body arguments");
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (argument.getType() != input.getType())
      return reject(task,
                    "HOST_LOWERING_BODY_CONTRACT: body argument type does "
                    "not match its task input");

  auto yield = dyn_cast<TaskflowYieldOp>(body.getTerminator());
  if (!yield)
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: task must end in "
                        "taskflow.yield");
  if (yield.getDoneReads().size() != task.getDoneReads().size() ||
      yield.getDoneWrites().size() != task.getDoneWrites().size() ||
      yield.getValueResults().size() != task.getValueOutputs().size())
    return reject(yield,
                  "HOST_LOWERING_BODY_CONTRACT: yield/result segment count "
                  "mismatch");
  auto checkTypes = [&](ValueRange values, ValueRange results) {
    for (auto [value, result] : llvm::zip(values, results))
      if (value.getType() != result.getType())
        return failure();
    return success();
  };
  if (failed(checkTypes(yield.getDoneReads(), task.getDoneReads())) ||
      failed(checkTypes(yield.getDoneWrites(), task.getDoneWrites())) ||
      failed(checkTypes(yield.getValueResults(), task.getValueOutputs())))
    return reject(yield,
                  "HOST_LOWERING_BODY_CONTRACT: yielded value type does not "
                  "match the task result");

  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  for (Operation &operation : body.without_terminator()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      if (failed(validateCounter(counter)))
        return failure();
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (hyperblock)
        return reject(candidate, "HOST_LOWERING_UNSUPPORTED_SHAPE: only one "
                                 "hyperblock per task is supported");
      hyperblock = candidate;
      continue;
    }
    if (isTaskflowOperation(&operation))
      return reject(&operation,
                    "HOST_LOWERING_UNSUPPORTED_OPERATION: unexpected "
                    "Taskflow operation in task body");
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp>(&operation))
      continue;
    for (Region &region : operation.getRegions())
      if (failed(validateHostRegion(region, &operation)))
        return failure();
    if (!isSupportedHostOperation(&operation))
      return reject(&operation, "HOST_LOWERING_UNSUPPORTED_OPERATION: " +
                                    operation.getName().getStringRef());
  }
  if (!counters.empty() && !hyperblock)
    return reject(task, "HOST_LOWERING_UNSUPPORTED_SHAPE: counters require a "
                        "hyperblock");
  if (hyperblock && failed(validateHyperblock(hyperblock, counters)))
    return failure();
  return success();
}

static void collectTaskDependencies(
    Value value, const llvm::DenseMap<Operation *, unsigned> &taskIndices,
    llvm::DenseSet<unsigned> &dependencies) {
  if (auto task = value.getDefiningOp<TaskflowTaskOp>()) {
    auto found = taskIndices.find(task.getOperation());
    if (found != taskIndices.end())
      dependencies.insert(found->second);
    return;
  }
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    collectTaskDependencies(channel.getSource(), taskIndices, dependencies);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates())
      collectTaskDependencies(state, taskIndices, dependencies);
  }
}

static FailureOr<SmallVector<TaskflowTaskOp>>
dependencyOrder(func::FuncOp function) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  llvm::DenseMap<Operation *, unsigned> taskIndices;
  for (auto [index, task] : llvm::enumerate(tasks))
    taskIndices[task.getOperation()] = index;

  SmallVector<llvm::SmallDenseSet<unsigned>> dependencies(tasks.size());
  for (auto [index, task] : llvm::enumerate(tasks)) {
    llvm::DenseSet<unsigned> found;
    for (Value value : task.getWillReads())
      collectTaskDependencies(value, taskIndices, found);
    for (Value value : task.getWillWrites())
      collectTaskDependencies(value, taskIndices, found);
    for (Value value : task.getValueInputs())
      collectTaskDependencies(value, taskIndices, found);
    found.erase(index);
    dependencies[index].insert(found.begin(), found.end());
  }

  SmallVector<unsigned> state(tasks.size(), 0);
  SmallVector<TaskflowTaskOp> order;
  std::function<LogicalResult(unsigned)> visit = [&](unsigned index) {
    if (state[index] == 2)
      return success();
    if (state[index] == 1)
      return reject(tasks[index],
                    "HOST_LOWERING_DEPENDENCY_CYCLE: task dependencies are "
                    "not acyclic");
    state[index] = 1;
    for (unsigned dependency : dependencies[index])
      if (failed(visit(dependency)))
        return failure();
    state[index] = 2;
    order.push_back(tasks[index]);
    return success();
  };
  for (unsigned index = 0; index < tasks.size(); ++index)
    if (failed(visit(index)))
      return failure();
  return order;
}

static FailureOr<Value> mappedValue(IRMapping &mapping, Value value,
                                    Operation *owner) {
  Value mapped = mapping.lookupOrNull(value);
  if (!mapped)
    return reject(owner,
                  "HOST_LOWERING_BODY_CONTRACT: body value was not mapped "
                  "into the host function");
  return mapped;
}

static LogicalResult lowerHyperblock(TaskflowHyperblockOp hyperblock,
                                     IRMapping &mapping, OpBuilder &builder,
                                     Operation *&completionSite) {
  Block &sourceBody = hyperblock.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  for (Value index : hyperblock.getIndices()) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    if (!counter)
      return reject(hyperblock,
                    "HOST_LOWERING_BODY_CONTRACT: missing counter definition");
    counters.push_back(counter);
  }

  OpBuilder::InsertionGuard guard(builder);
  // The active insertion point is the parent task in the host function.
  // Creating loops inside the source task region would erase them with it.
  SmallVector<scf::ForOp> loops;
  for (TaskflowCounterOp counter : counters) {
    FailureOr<Value> lower =
        mappedValue(mapping, counter.getLowerBound(), counter.getOperation());
    FailureOr<Value> upper =
        mappedValue(mapping, counter.getUpperBound(), counter.getOperation());
    FailureOr<Value> step =
        mappedValue(mapping, counter.getStep(), counter.getOperation());
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    scf::ForOp loop =
        builder.create<scf::ForOp>(hyperblock.getLoc(), *lower, *upper, *step);
    loops.push_back(loop);
    builder.setInsertionPointToStart(loop.getBody());
    mapping.map(counter.getCounterIndex(), loop.getInductionVar());
  }
  if (loops.empty())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock has no "
                  "counter loop");

  for (auto [argument, loop] : llvm::zip(sourceBody.getArguments(), loops))
    mapping.map(argument, loop.getInductionVar());

  builder.setInsertionPointToStart(loops.back().getBody());
  for (Operation &operation : sourceBody.without_terminator()) {
    for (Value operand : operation.getOperands())
      if (!mapping.lookupOrNull(operand))
        return reject(&operation,
                      "HOST_LOWERING_BODY_CONTRACT: hyperblock operand was "
                      "not mapped");
    builder.clone(operation, mapping);
  }
  // The body of every nested loop needs its own terminator.  The innermost
  // loop already contains the cloned hyperblock body and receives its yield
  // first; outer loops only contain the next inner loop and need an additional
  // yield after it.
  for (scf::ForOp loop : loops) {
    if (loop.getBody()->getTerminator())
      continue;
    OpBuilder::InsertionGuard yieldGuard(builder);
    builder.setInsertionPointToEnd(loop.getBody());
    builder.create<scf::YieldOp>(hyperblock.getLoc());
  }
  completionSite = loops.front().getOperation();
  hyperblock.erase();
  return success();
}

static void markCompletionSite(TaskflowTaskOp task, Operation *completionSite,
                               OpBuilder &builder) {
  if (!completionSite)
    return;
  completionSite->setAttr(kHostTaskNameAttr,
                          builder.getStringAttr(task.getTaskName()));
  completionSite->setAttr(kHostTaskCompletionAttr,
                          UnitAttr::get(task.getContext()));
  // Preserve semantic provenance at the host completion site.  This lets a
  // host interpreter observe partial buffers, state-carry blocks, and numeric
  // reduction tasks without retaining a non-executable Taskflow marker op.
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.starts_with("amoeba.semantic.") ||
        name.starts_with("amoeba.tiling."))
      completionSite->setAttr(name, attribute.getValue());
  }
}

static LogicalResult lowerTask(TaskflowTaskOp task, unsigned ordinal,
                               bool captureTaskStates) {
  Block &taskBody = task.getBody().front();
  SmallVector<Value> inputs;
  llvm::append_range(inputs, task.getWillReads());
  llvm::append_range(inputs, task.getWillWrites());
  llvm::append_range(inputs, task.getValueInputs());

  IRMapping mapping;
  for (auto [argument, input] : llvm::zip(taskBody.getArguments(), inputs))
    mapping.map(argument, input);

  OpBuilder builder(task.getContext());
  builder.setInsertionPoint(task);
  TaskflowYieldOp yield = cast<TaskflowYieldOp>(taskBody.getTerminator());
  Operation *completionSite = nullptr;
  SmallVector<Operation *> bodyOperations;
  for (Operation &operation : taskBody.without_terminator())
    bodyOperations.push_back(&operation);
  for (Operation *operation : bodyOperations) {
    if (isa<TaskflowCounterOp>(operation))
      continue;
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(operation)) {
      if (failed(lowerHyperblock(hyperblock, mapping, builder, completionSite)))
        return failure();
      builder.setInsertionPoint(task);
      continue;
    }
    for (Value operand : operation->getOperands())
      if (!mapping.lookupOrNull(operand))
        return reject(operation,
                      "HOST_LOWERING_BODY_CONTRACT: task operand was not "
                      "mapped into the host function");
    completionSite = builder.clone(*operation, mapping);
    builder.setInsertionPoint(task);
  }

  SmallVector<Value> replacements;
  auto appendMapped = [&](ValueRange values) -> LogicalResult {
    for (Value value : values) {
      FailureOr<Value> mapped = mappedValue(mapping, value, task);
      if (failed(mapped))
        return failure();
      replacements.push_back(*mapped);
    }
    return success();
  };
  if (failed(appendMapped(yield.getDoneReads())) ||
      failed(appendMapped(yield.getDoneWrites())) ||
      failed(appendMapped(yield.getValueResults())))
    return failure();
  if (replacements.size() != task->getNumResults())
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: replacement result count "
                        "does not match task results");
  markCompletionSite(task, completionSite, builder);
  if (captureTaskStates) {
    if (!completionSite || yield.getDoneWrites().empty())
      return reject(task, "HOST_LOWERING_TRACE_UNSUPPORTED: task has no "
                          "completion site or written state");
    ModuleOp module = task->getParentOfType<ModuleOp>();
    auto function = task->getParentOfType<func::FuncOp>();
    OpBuilder traceBuilder(task.getContext());
    traceBuilder.setInsertionPointAfter(completionSite);
    for (auto [writeIndex, written] : llvm::enumerate(yield.getDoneWrites())) {
      Value state = mapping.lookupOrNull(written);
      auto type =
          dyn_cast_or_null<MemRefType>(state ? state.getType() : Type());
      if (!type || !type.hasStaticShape())
        return reject(task, "HOST_LOWERING_TRACE_UNSUPPORTED: written state "
                            "must be a static memref");
      std::string symbol = ("__orbit_trace_" + function.getSymName() + "_" +
                            Twine(ordinal) + "_" + Twine(writeIndex))
                               .str();
      OpBuilder globalBuilder(task.getContext());
      globalBuilder.setInsertionPointToStart(module.getBody());
      auto global = globalBuilder.create<memref::GlobalOp>(
          task.getLoc(), globalBuilder.getStringAttr(symbol), StringAttr(),
          TypeAttr::get(type), UnitAttr::get(task.getContext()), UnitAttr(),
          IntegerAttr());
      auto snapshot = traceBuilder.create<memref::GetGlobalOp>(
          task.getLoc(), type, global.getName());
      auto copy = traceBuilder.create<memref::CopyOp>(task.getLoc(), state,
                                                      snapshot.getResult());
      copy->setAttr(kHostTraceSymbolAttr, traceBuilder.getStringAttr(symbol));
      copy->setAttr("amoeba.host.trace_task_index",
                    traceBuilder.getI64IntegerAttr(ordinal));
      copy->setAttr("amoeba.host.trace_write_index",
                    traceBuilder.getI64IntegerAttr(writeIndex));
      traceBuilder.setInsertionPointAfter(copy);
    }
  }
  task->replaceAllUsesWith(replacements);
  task.erase();
  return success();
}

static LogicalResult eliminateAliases(func::FuncOp function) {
  SmallVector<TaskflowJoinOp> joins;
  SmallVector<TaskflowChannelOp> channels;
  function.walk([&](TaskflowJoinOp join) { joins.push_back(join); });
  function.walk(
      [&](TaskflowChannelOp channel) { channels.push_back(channel); });

  // Verification is intentionally done before the first alias rewrite.  A
  // completion join is only an alias after the existing Taskflow verifier has
  // proved exact, disjoint region coverage.
  for (TaskflowJoinOp join : joins)
    if (failed(verify(join.getOperation())))
      return reject(join,
                    "HOST_LOWERING_UNVERIFIED_JOIN: completion join proof "
                    "did not verify");
  for (TaskflowJoinOp join : joins) {
    join.getJoined().replaceAllUsesWith(join.getBase());
    join.erase();
  }
  for (TaskflowChannelOp channel : channels) {
    channel.getTarget().replaceAllUsesWith(channel.getSource());
    channel.erase();
  }
  return success();
}

static LogicalResult validateFunction(func::FuncOp function) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  for (TaskflowTaskOp task : tasks)
    if (failed(validateTask(task)))
      return failure();
  // Verify aliases before lowering.  This also diagnoses malformed joins and
  // channels without relying on a later pass to notice the problem.
  bool valid = true;
  function.walk([&](TaskflowJoinOp join) {
    if (failed(verify(join.getOperation()))) {
      join.emitError() << kErrorPrefix << "HOST_LOWERING_UNVERIFIED_JOIN: "
                       << "completion join proof did not verify";
      valid = false;
    }
  });
  return valid ? success() : failure();
}

static LogicalResult lowerFunction(func::FuncOp function,
                                   bool captureTaskStates) {
  if (failed(validateFunction(function)))
    return failure();
  FailureOr<SmallVector<TaskflowTaskOp>> order = dependencyOrder(function);
  if (failed(order))
    return failure();
  if (failed(eliminateAliases(function)))
    return failure();
  for (auto [ordinal, task] : llvm::enumerate(*order))
    if (failed(lowerTask(task, ordinal, captureTaskStates)))
      return failure();

  bool leftover = false;
  function.walk([&](Operation *operation) {
    if (isTaskflowOperation(operation))
      leftover = true;
  });
  if (leftover)
    return reject(function,
                  "HOST_LOWERING_UNSUPPORTED_OPERATION: Taskflow operation "
                  "remains after lowering");
  return verify(function.getOperation());
}

struct LowerJointTaskflowToHostSCFPass
    : public PassWrapper<LowerJointTaskflowToHostSCFPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJointTaskflowToHostSCFPass)

  LowerJointTaskflowToHostSCFPass() = default;
  LowerJointTaskflowToHostSCFPass(const LowerJointTaskflowToHostSCFPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "lower-joint-taskflow-to-host-scf";
  }
  StringRef getDescription() const override {
    return "Lower executable ORBIT Taskflow graphs to host SCF";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Only lower this function; empty lowers every Taskflow "
                     "function."),
      llvm::cl::init("")};
  Option<bool> captureTaskStates{
      *this, "capture-task-states",
      llvm::cl::desc("Snapshot each task's written memrefs for host numeric "
                     "auditing"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) {
      if (functionName.getValue().empty() ||
          function.getSymName() == functionName.getValue()) {
        bool hasTaskflow = false;
        function.walk([&](Operation *operation) {
          if (isTaskflowOperation(operation))
            hasTaskflow = true;
        });
        if (hasTaskflow)
          functions.push_back(function);
      }
    });
    if (!functionName.getValue().empty() && functions.empty()) {
      module.emitError() << kErrorPrefix << "HOST_LOWERING_FUNCTION_NOT_FOUND: "
                         << "no Taskflow function named "
                         << functionName.getValue();
      return signalPassFailure();
    }
    for (func::FuncOp function : functions)
      if (failed(lowerFunction(function, captureTaskStates)))
        return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createLowerJointTaskflowToHostSCFPass() {
  return std::make_unique<LowerJointTaskflowToHostSCFPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
