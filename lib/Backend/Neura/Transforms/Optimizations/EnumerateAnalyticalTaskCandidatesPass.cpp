//===- EnumerateAnalyticalTaskCandidatesPass.cpp -------------------------===//
//
// Implements the pass that freezes the complete rectangular shape space.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct EnumerateAnalyticalTaskCandidatesPass
    : public PassWrapper<EnumerateAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateAnalyticalTaskCandidatesPass)

  EnumerateAnalyticalTaskCandidatesPass() = default;
  EnumerateAnalyticalTaskCandidatesPass(
      const EnumerateAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Freezes every rectangular task-shape candidate without scoring or "
           "running the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Candidate JSONL output path."),
                                 llvm::cl::init("")};
  Option<int64_t> maxCandidates{
      *this, "max-candidates",
      llvm::cl::desc("Fails rather than publishing a partial manifest."),
      llvm::cl::init(1000000)};
  Option<int64_t> maxCgrasPerTask{
      *this, "max-cgras-per-task",
      llvm::cl::desc("Limits the rectangular physical footprint per task."),
      llvm::cl::init(4)};

  void runOnOperation() override {
    // Selects the Taskflow function and rejects invalid safety limits. This
    // pass observes the IR; its only output is the candidate JSONL file.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (outputFile.getValue().empty() || maxCandidates.getValue() <= 0 ||
        maxCgrasPerTask.getValue() <= 0) {
      func.emitError() << "output, positive max-candidates, and positive "
                          "max-cgras-per-task are required";
      return signalPassFailure();
    }

    // Snapshots the semantic task facts and builds the single-task shape
    // alphabet from the physical architecture.
    FailureOr<SmallVector<TaskFact>> taskFacts = collectTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    FailureOr<std::string> architectureFingerprint =
        currentArchitectureFingerprint(error);
    if (failed(architectureFingerprint)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    SmallVector<RectShape> shapes = enumerateRectShapes(
        architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
        architecture.getPerCgraRows(), architecture.getPerCgraColumns(),
        maxCgrasPerTask.getValue());
    if (shapes.empty()) {
      func.emitError() << "declared rectangular shape space is empty";
      return signalPassFailure();
    }

    // Computes the complete program space as S^T for T ordered tasks. Each
    // task must fit by itself because later temporal scheduling may reuse the
    // same CGRAs. The limit rejects the entire space instead of truncating it.
    uint64_t candidateCount = 1;
    for (size_t ignored = 0; ignored < taskFacts->size(); ++ignored) {
      if (candidateCount >
          static_cast<uint64_t>(maxCandidates.getValue()) / shapes.size()) {
        func.emitError()
            << "complete shape space exceeds max-candidates="
            << maxCandidates.getValue()
            << "; refusing to publish a partial candidate manifest";
        return signalPassFailure();
      }
      candidateCount *= shapes.size();
    }

    const std::string function = func.getSymName().str();
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          // Freezes every input needed to reconstruct the candidate space. The
          // cost-query list also de-duplicates the requests made to the model.
          llvm::json::Object architectureRecord;
          architectureRecord["grid_rows"] =
              int64_t{architecture.getMultiCgraRows()};
          architectureRecord["grid_cols"] =
              int64_t{architecture.getMultiCgraColumns()};
          architectureRecord["per_cgra_tile_rows"] =
              int64_t{architecture.getPerCgraRows()};
          architectureRecord["per_cgra_tile_cols"] =
              int64_t{architecture.getPerCgraColumns()};
          architectureRecord["spec_fingerprint"] = *architectureFingerprint;

          llvm::json::Array tasks;
          for (const TaskFact &task : *taskFacts) {
            llvm::json::Object record;
            record["task"] = task.name;
            record["body_id"] = task.bodyId;
            record["trip_count"] = task.tripCount;
            tasks.push_back(std::move(record));
          }
          llvm::json::Array costQueries;
          for (const TaskFact &task : *taskFacts) {
            for (const RectShape &shape : shapes) {
              llvm::json::Object query;
              query["task"] = task.name;
              query["body_id"] = task.bodyId;
              query["mapper_shape_id"] = shape.mapperShapeId;
              query["mapper_tile_rows"] = shape.mapperRows;
              query["mapper_tile_cols"] = shape.mapperCols;
              costQueries.push_back(std::move(query));
            }
          }
          llvm::json::Object fixedAxes;
          fixedAxes["fusion"] = "identity";
          fixedAxes["fission"] = "factor-1";
          fixedAxes["tiling"] = "factor-1";
          fixedAxes["placement"] = "downstream-heuristic";
          fixedAxes["temporal_order"] = "downstream-heuristic";
          fixedAxes["communication"] = "not-scored";
          llvm::json::Object header;
          header["record_type"] = "header";
          header["schema_version"] = kCandidateSchema.str();
          header["search_scope"] = kSearchScope.str();
          header["shape_policy"] = kShapePolicy.str();
          header["candidate_identity"] = kCandidateIdentity.str();
          header["function"] = function;
          header["architecture"] = std::move(architectureRecord);
          header["max_cgras_per_task"] = maxCgrasPerTask.getValue();
          header["tasks"] = std::move(tasks);
          header["cost_queries"] = std::move(costQueries);
          header["fixed_axes"] = std::move(fixedAxes);
          writeJsonLine(os, std::move(header));

          // Emits the Cartesian product in task-major mixed-radix order. This
          // pass has no analytical score, so it retains every valid shape.
          llvm::SHA256 idsDigest;
          uint64_t emitted = 0;
          SmallVector<TaskShapeChoice> selected;
          std::function<void(size_t)> visit = [&](size_t taskIndex) {
            if (taskIndex == taskFacts->size()) {
              Candidate candidate;
              candidate.choices = selected;
              candidate.id = makeCandidateId(function, *architectureFingerprint,
                                             candidate.choices);
              writeJsonLine(os, candidateJson(candidate));
              updateCandidateIdDigest(idsDigest, candidate.id);
              ++emitted;
              return;
            }
            const TaskFact &task = (*taskFacts)[taskIndex];
            for (const RectShape &shape : shapes) {
              selected.push_back(
                  {task.name, task.bodyId, task.tripCount, shape});
              visit(taskIndex + 1);
              selected.pop_back();
            }
          };
          visit(0);
          if (emitted != candidateCount) {
            error = "internal candidate-count mismatch";
            return false;
          }

          // Closes the stream with its record count and ordered-ID digest. The
          // common reader recomputes both values before scoring or replay.
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema_version"] = kCandidateSchema.str();
          footer["candidate_count"] = static_cast<int64_t>(emitted);
          footer["candidate_ids_sha256"] =
              llvm::toHex(idsDigest.final(), /*LowerCase=*/true);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }
    llvm::errs() << "[AnalyticalTaskDSE] enumerated " << candidateCount
                 << " complete shape candidates into " << outputFile.getValue()
                 << "\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass() {
  return std::make_unique<EnumerateAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
