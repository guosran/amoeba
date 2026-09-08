//===- ScoreAnalyticalTaskCandidatesPass.cpp -----------------------------===//
//
// Implements full-candidate scoring and deterministic top-k selection.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct ScoreAnalyticalTaskCandidatesPass
    : public PassWrapper<ScoreAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ScoreAnalyticalTaskCandidatesPass)

  ScoreAnalyticalTaskCandidatesPass() = default;
  ScoreAnalyticalTaskCandidatesPass(
      const ScoreAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "score-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Scores every frozen candidate through a shared task-shape ML "
           "cost cache, then emits top-k";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> costFile{
      *this, "cost-file", llvm::cl::desc("Task-shape cost catalogue path."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Score JSONL output path."),
                                 llvm::cl::init("")};
  Option<int64_t> topK{
      *this, "top-k",
      llvm::cl::desc("Uses zero to select every valid candidate."),
      llvm::cl::init(1)};

  void runOnOperation() override {
    // Establishes that the IR, candidate input, model catalogue, and score
    // output are distinct and refer to the same function and architecture.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (candidateFile.getValue().empty() || costFile.getValue().empty() ||
        outputFile.getValue().empty() || topK.getValue() < 0 ||
        samePath(candidateFile.getValue(), outputFile.getValue()) ||
        samePath(costFile.getValue(), outputFile.getValue()) ||
        samePath(candidateFile.getValue(), costFile.getValue())) {
      func.emitError() << "distinct candidates, cost-file, and output paths "
                          "and non-negative top-k are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskFact>> taskFacts = collectTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    if (!costs.load(costFile.getValue(), func.getSymName(), *taskFacts,
                    error)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Retains every supported score until the manifest reader reaches and
    // verifies its footer. Every candidate receives a score record first.
    SmallVector<RankedCandidate> ranked;
    ManifestHeader manifestHeader;
    ManifestFooter manifestFooter;
    uint64_t scoredCount = 0;
    uint64_t validCount = 0;
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          // Records the schema, model namespace, and score model that produced
          // this ranking. The candidate reader separately validates the
          // architecture dimensions and complete candidate space.
          llvm::json::Object scoreHeader;
          scoreHeader["record_type"] = "header";
          scoreHeader["schema"] = kScoreSchema.str();
          scoreHeader["candidate_schema"] = kCandidateSchema.str();
          scoreHeader["function"] = func.getSymName().str();
          scoreHeader["cost_namespace"] = costs.nameSpace().str();
          scoreHeader["candidate_manifest_sha256"] =
              costs.candidateManifestSha256().str();
          scoreHeader["architecture_sha256"] = costs.architectureSha256().str();
          scoreHeader["score_model"] = kScoreModel.str();
          scoreHeader["mapper_success_probability"] = "diagnostic_only";
          writeJsonLine(os, std::move(scoreHeader));

          auto consume = [&](uint64_t manifestIndex, const Candidate &candidate,
                             std::string &consumeError) {
            // Computes one shape-only program score as follows.
            //
            //   duration(task) = startup + II * (trip_count - 1)
            //   score(candidate) = max duration(task)
            //
            // Represents only a compute bottleneck until later search scopes
            // add communication and temporal scheduling.
            bool valid = true;
            double bottleneck = 0.0;
            std::string rejectReason;
            llvm::json::Array taskCosts;
            for (const TaskShapeChoice &choice : candidate.choices) {
              const TaskShapeCost *cost = costs.get(choice, consumeError);
              if (!cost)
                return false;
              llvm::json::Object taskCost;
              taskCost["task"] = choice.task;
              taskCost["mapper_tile_rows"] = choice.shape.mapperRows;
              taskCost["mapper_tile_cols"] = choice.shape.mapperCols;
              taskCost["predicted_ii"] = cost->predictedII;
              taskCost["startup_cycles"] = cost->startupCycles;
              taskCost["trip_count"] = choice.tripCount;
              if (!cost->supported) {
                valid = false;
                if (rejectReason.empty())
                  rejectReason = "UNSUPPORTED_TASK_SHAPE";
                taskCost["support_status"] = "unsupported";
              } else {
                double duration = cost->startupCycles +
                                  cost->predictedII *
                                      static_cast<double>(choice.tripCount - 1);
                if (!std::isfinite(duration)) {
                  consumeError =
                      "task duration overflow for task=" + choice.task +
                      ", mapper_shape=rect-" +
                      std::to_string(choice.shape.mapperRows) + "x" +
                      std::to_string(choice.shape.mapperCols);
                  return false;
                }
                taskCost["support_status"] = "supported";
                taskCost["predicted_duration"] = duration;
                bottleneck = std::max(bottleneck, duration);
              }
              taskCosts.push_back(std::move(taskCost));
            }

            llvm::json::Object score;
            score["record_type"] = "score";
            score["schema"] = kScoreSchema.str();
            score["candidate_id"] = candidate.id;
            score["valid"] = valid;
            score["task_costs"] = std::move(taskCosts);
            if (valid) {
              score["predicted_compute_bottleneck"] = bottleneck;
              ranked.push_back({candidate.id, manifestIndex, bottleneck});
              ++validCount;
            } else {
              score["reject_reason"] = rejectReason;
            }
            writeJsonLine(os, std::move(score));
            ++scoredCount;
            return true;
          };

          // Invokes the consumer once per canonical record and rejects an
          // incomplete, reordered, or tampered candidate space.
          if (!readCandidateManifest(
                  candidateFile.getValue(), *taskFacts, func.getSymName(),
                  ::mlir::neura::getArchitecture(),
                  costs.candidateManifestSha256(), costs.architectureSha256(),
                  consume, manifestHeader, manifestFooter, error))
            return false;
          if (scoredCount != manifestFooter.candidateCount) {
            error = "not every frozen candidate was scored";
            return false;
          }
          // Enumeration declares only (task, shape) queries used by at least
          // one concurrently packable candidate. Full traversal must therefore
          // touch every catalogue entry. Equality rejects stale extras; a
          // missing entry fails at lookup above.
          if (costs.entries() != costs.catalogEntries()) {
            error = "cost catalogue does not exactly cover candidate manifest "
                    "task-shape queries";
            return false;
          }

          // Sorts only after the proven manifest count has been scored. The
          // numeric manifest index preserves enumeration order when scores tie;
          // comparing strings would incorrectly place candidate-13 before
          // candidate-5.
          llvm::sort(ranked, [](const RankedCandidate &lhs,
                                const RankedCandidate &rhs) {
            if (lhs.score != rhs.score)
              return lhs.score < rhs.score;
            return lhs.manifestIndex < rhs.manifestIndex;
          });
          uint64_t selected =
              topK.getValue() == 0
                  ? ranked.size()
                  : std::min<uint64_t>(topK.getValue(), ranked.size());
          llvm::json::Array shortlist;
          for (uint64_t index = 0; index < selected; ++index) {
            llvm::json::Object item;
            item["rank"] = static_cast<int64_t>(index);
            item["candidate_id"] = ranked[index].id;
            item["predicted_compute_bottleneck"] = ranked[index].score;
            shortlist.push_back(std::move(item));
          }

          // Directs the external driver to materialize only the shortlist. It
          // also records enough counts to audit full scoring.
          llvm::json::Object cacheStats;
          cacheStats["hits"] = static_cast<int64_t>(costs.hits());
          cacheStats["misses"] = static_cast<int64_t>(costs.misses());
          cacheStats["entries"] = static_cast<int64_t>(costs.entries());
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema"] = kScoreSchema.str();
          footer["candidate_count"] =
              static_cast<int64_t>(manifestFooter.candidateCount);
          footer["scored_count"] = static_cast<int64_t>(scoredCount);
          footer["valid_count"] = static_cast<int64_t>(validCount);
          footer["top_k_requested"] = topK.getValue();
          footer["shortlist"] = std::move(shortlist);
          footer["cache"] = std::move(cacheStats);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }
    llvm::errs() << "[AnalyticalTaskDSE] scored all " << scoredCount
                 << " candidates with " << costs.entries()
                 << " unique task-shape cost queries\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createScoreAnalyticalTaskCandidatesPass() {
  return std::make_unique<ScoreAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
