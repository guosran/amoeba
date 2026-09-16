//===- MapJointSchedulingTasksPass.cpp ----------------------------------===//
//
// Replays Neura mapping independently for each selected task rectangle.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendPasses.h"

#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <cmath>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct MapJointSchedulingTasksPass
    : public PassWrapper<MapJointSchedulingTasksPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapJointSchedulingTasksPass)

  MapJointSchedulingTasksPass() = default;
  MapJointSchedulingTasksPass(const MapJointSchedulingTasksPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "map-joint-scheduling-tasks";
  }
  StringRef getDescription() const override {
    return "Map each selected task on its own fixed oriented rectangle";
  }

  Option<std::string> scoreFile{*this, "scores",
                                llvm::cl::desc("Validated score JSONL path."),
                                llvm::cl::init("")};
  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Unique Taskflow function selected by the manifest."),
      llvm::cl::init("")};
  Option<std::string> candidateId{*this, "candidate-id",
                                  llvm::cl::desc("Selected candidate ID."),
                                  llvm::cl::init("")};
  Option<std::string> expectedTraceSha256{
      *this, "expected-trace-sha256",
      llvm::cl::desc("Expected selected schedule trace identity."),
      llvm::cl::init("")};
  Option<std::string> expectedScoreSha256{
      *this, "expected-score-sha256",
      llvm::cl::desc("SHA-256 of the exact validated score JSONL."),
      llvm::cl::init("")};

  llvm::StringMap<double> startupCycles;

  LogicalResult loadStartupCycles(ModuleOp module, StringRef function) {
    if (scoreFile.getValue().empty() || candidateId.getValue().empty() ||
        expectedTraceSha256.getValue().empty() ||
        expectedScoreSha256.getValue().empty())
      return module.emitError("per-task mapper replay requires scores and "
                              "candidate-id");
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(scoreFile.getValue());
    if (!buffer)
      return module.emitError("cannot read score JSONL for mapper replay");
    llvm::SHA256 hasher;
    hasher.update((*buffer)->getBuffer());
    if (llvm::toHex(hasher.final(), /*LowerCase=*/true) !=
        expectedScoreSha256.getValue())
      return module.emitError("score JSONL byte identity mismatch");
    bool found = false;
    bool sawHeader = false;
    bool sawFooter = false;
    uint64_t scoreCount = 0;
    std::optional<int64_t> declaredScoreCount;
    for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
         !lines.is_at_end(); ++lines) {
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
      if (!parsed)
        return module.emitError("invalid score JSONL for mapper replay");
      llvm::json::Object *object = parsed->getAsObject();
      auto recordType =
          object ? object->getString("record_type") : std::nullopt;
      auto schema = object ? object->getString("schema") : std::nullopt;
      if (!recordType || !schema ||
          *schema != ::mlir::amoeba::neura::joint_scheduling::kScoreSchema)
        return module.emitError("score JSONL contract is invalid");
      if (*recordType == "header") {
        if (sawHeader || sawFooter)
          return module.emitError("score JSONL header is misplaced");
        auto scoredFunction = object->getString("function");
        if (!scoredFunction || *scoredFunction != function)
          return module.emitError("score JSONL function identity mismatch");
        sawHeader = true;
        continue;
      }
      if (*recordType == "footer") {
        if (!sawHeader || sawFooter)
          return module.emitError("score JSONL footer is misplaced");
        sawFooter = true;
        declaredScoreCount = object->getInteger("scored_count");
        if (!declaredScoreCount || *declaredScoreCount < 0)
          return module.emitError("score JSONL footer count is invalid");
        continue;
      }
      if (*recordType != "score")
        return module.emitError("score JSONL record type is unsupported");
      if (!sawHeader || sawFooter)
        return module.emitError("score record is misplaced");
      ++scoreCount;
      if (object->getString("candidate_id") != candidateId.getValue())
        continue;
      if (found)
        return module.emitError("duplicate selected score record");
      std::optional<bool> valid = object->getBoolean("valid");
      llvm::json::Array *costs = object->getArray("task_costs");
      if (!valid || !*valid || !costs)
        return module.emitError("selected score record is not valid");
      auto traceSha = object->getString("schedule_trace_sha256");
      if (!traceSha || *traceSha != expectedTraceSha256.getValue())
        return module.emitError("selected schedule trace identity mismatch");
      for (llvm::json::Value &value : *costs) {
        llvm::json::Object *cost = value.getAsObject();
        std::optional<StringRef> task =
            cost ? cost->getString("task") : std::nullopt;
        std::optional<double> startup =
            cost ? cost->getNumber("startup_cycles") : std::nullopt;
        if (!task || !startup || !std::isfinite(*startup) || *startup < 0 ||
            !startupCycles.try_emplace(*task, *startup).second)
          return module.emitError("selected task startup cost is invalid");
      }
      found = true;
    }
    if (!found || !sawHeader || !sawFooter || !declaredScoreCount ||
        static_cast<uint64_t>(*declaredScoreCount) != scoreCount)
      return module.emitError("score JSONL is incomplete or selected record "
                              "is absent");
    return success();
  }

  LogicalResult mapTask(TaskflowTaskOp task) {
    if (!task->hasAttr("amoeba.joint_shape_orientation_fixed"))
      return task.emitError("task has no fixed joint-scheduling shape");
    auto mapper_rows =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_rows");
    auto mapper_cols =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_cols");
    if (!mapper_rows || !mapper_cols || mapper_rows.getInt() <= 0 ||
        mapper_cols.getInt() <= 0 ||
        mapper_rows.getInt() > std::numeric_limits<int>::max() ||
        mapper_cols.getInt() > std::numeric_limits<int>::max())
      return task.emitError("selected mapper rectangle is missing or invalid");

    SmallVector<::mlir::neura::KernelOp> kernels;
    task.walk(
        [&](::mlir::neura::KernelOp kernel) { kernels.push_back(kernel); });
    if (kernels.size() != 1)
      return task.emitError("per-task mapper replay requires exactly one "
                            "neura.kernel");
    ::mlir::neura::KernelOp kernel = kernels.front();
    if (kernel.getBody().empty())
      return task.emitError("cannot map an empty neura.kernel");

    MLIRContext *context = task.getContext();
    Location location = task.getLoc();
    ModuleOp temporary = ModuleOp::create(location);
    OpBuilder builder(context);
    builder.setInsertionPointToStart(temporary.getBody());
    SmallVector<Type> argument_types;
    for (BlockArgument argument : kernel.getBody().front().getArguments())
      argument_types.push_back(argument.getType());
    auto wrapper = builder.create<func::FuncOp>(
        location, "__joint_task_mapper_replay__",
        builder.getFunctionType(argument_types, kernel.getResultTypes()));
    wrapper->setAttr("accelerator", builder.getStringAttr("neura"));
    IRMapping mapping;
    kernel.getBody().cloneInto(&wrapper.getBody(), mapping);
    for (Block &block : wrapper.getBody()) {
      if (auto yield =
              dyn_cast<::mlir::neura::YieldOp>(block.getTerminator())) {
        builder.setInsertionPoint(yield);
        builder.create<func::ReturnOp>(location, yield.getResults());
        yield.erase();
      }
    }

    PassManager manager(context);
    manager.addPass(::mlir::neura::createInsertDataMovPass());
    ::mlir::neura::MapToAcceleratorOptions options;
    options.x_tiles = static_cast<int>(mapper_cols.getInt());
    options.y_tiles = static_cast<int>(mapper_rows.getInt());
    manager.addPass(::mlir::neura::createMapToAcceleratorPass(options));
    if (failed(manager.run(temporary)))
      return task.emitError("real per-task mapper replay failed");

    auto mapping_info = wrapper->getAttrOfType<DictionaryAttr>(
        ::mlir::neura::attr::kMappingInfo);
    auto compiled_ii =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kCompiledII)
            : IntegerAttr();
    auto mapped_rows =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kYTiles)
            : IntegerAttr();
    auto mapped_cols =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kXTiles)
            : IntegerAttr();
    if (!compiled_ii || compiled_ii.getInt() <= 0 || !mapped_rows ||
        !mapped_cols || mapped_rows.getInt() != mapper_rows.getInt() ||
        mapped_cols.getInt() != mapper_cols.getInt())
      return task.emitError(
          "mapper result does not match the selected oriented rectangle");

    // Preserve the actual mapped operations, not merely the mapper metadata.
    for (Block &block : wrapper.getBody()) {
      if (auto return_op = dyn_cast<func::ReturnOp>(block.getTerminator())) {
        builder.setInsertionPoint(return_op);
        builder.create<::mlir::neura::YieldOp>(location, ValueRange{},
                                               return_op.getOperands());
        return_op.erase();
      }
    }
    kernel.getBody().takeBody(wrapper.getBody());

    task->setAttr("compiled_ii", compiled_ii);
    task->setAttr("amoeba.mapper_mapping_info", mapping_info);
    task->setAttr("amoeba.mapper_replay_verified", builder.getUnitAttr());
    auto tripCount =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_trip_count");
    auto startup = startupCycles.find(task.getTaskName());
    if (!tripCount || tripCount.getInt() <= 0 || startup == startupCycles.end())
      return task.emitError("mapper replay has no bound trip/startup cost");
    const long double duration =
        static_cast<long double>(startup->second) +
        static_cast<long double>(compiled_ii.getInt()) *
            static_cast<long double>(tripCount.getInt() - 1);
    if (!std::isfinite(duration) || duration < 1 ||
        duration > std::numeric_limits<int64_t>::max())
      return task.emitError("real mapped task duration exceeds int64");
    const int64_t roundedDuration = static_cast<int64_t>(std::ceil(duration));
    SmallVector<NamedAttribute> profile;
    profile.push_back(builder.getNamedAttr(
        "duration", builder.getI64IntegerAttr(roundedDuration)));
    task->setAttr("profile_info", builder.getDictionaryAttr(profile));
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        ::mlir::amoeba::neura::joint_scheduling::selectTaskFunction(
            module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function == *selectedFunction)
        continue;
      bool hasTasks = false;
      function.walk([&](TaskflowTaskOp) { hasTasks = true; });
      if (hasTasks) {
        function.emitError("mapper replay rejects additional task-bearing "
                           "functions outside its selected function");
        return signalPassFailure();
      }
    }
    if (failed(loadStartupCycles(module, selectedFunction->getSymName())))
      return signalPassFailure();
    SmallVector<TaskflowTaskOp> tasks;
    selectedFunction->walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
    if (tasks.empty()) {
      module.emitError("per-task mapper replay requires taskflow tasks");
      return signalPassFailure();
    }
    for (TaskflowTaskOp task : tasks) {
      auto function = task->getParentOfType<func::FuncOp>();
      auto selected = function ? function->getAttrOfType<StringAttr>(
                                     "joint_scheduling_candidate_id")
                               : StringAttr();
      if (!selected || selected.getValue() != candidateId.getValue()) {
        task.emitError("mapper replay candidate identity does not match IR");
        return signalPassFailure();
      }
    }
    if (startupCycles.size() != tasks.size()) {
      module.emitError("selected score task coverage does not match replay");
      return signalPassFailure();
    }
    for (TaskflowTaskOp task : tasks)
      if (failed(mapTask(task)))
        return signalPassFailure();
    (*selectedFunction)
        ->setAttr("joint_scheduling_mapper_replay_completed",
                  UnitAttr::get(module.getContext()));
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createMapJointSchedulingTasksPass() {
  return std::make_unique<MapJointSchedulingTasksPass>();
}

} // namespace mlir::amoeba::neura
