//===- MaterializeAnalyticalTaskCandidatePass.cpp ------------------------===//
//
// Implements replay of one validated shape candidate onto Taskflow IR.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct MaterializeAnalyticalTaskCandidatePass
    : public PassWrapper<MaterializeAnalyticalTaskCandidatePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeAnalyticalTaskCandidatePass)

  MaterializeAnalyticalTaskCandidatePass() = default;
  MaterializeAnalyticalTaskCandidatePass(
      const MaterializeAnalyticalTaskCandidatePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-analytical-task-candidate";
  }
  StringRef getDescription() const override {
    return "Writes one selected shape candidate onto Taskflow IR without "
           "invoking the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> candidateIdOption{
      *this, "candidate-id", llvm::cl::desc("Candidate ID to materialize."),
      llvm::cl::init("")};
  Option<int64_t> candidateIndex{
      *this, "candidate-index",
      llvm::cl::desc("Selects a manifest index for testing or debugging."),
      llvm::cl::init(-1)};

  void runOnOperation() override {
    // Requires exactly one selector so a sequential ID and a debugging index
    // cannot disagree about which candidate should be materialized.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    bool hasId = !candidateIdOption.getValue().empty();
    bool hasIndex = candidateIndex.getValue() >= 0;
    if (candidateFile.getValue().empty() || hasId == hasIndex) {
      func.emitError()
          << "candidates and exactly one of candidate-id/candidate-index are "
             "required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskFact>> taskFacts =
        collectAnalyticalTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    // Continues through the footer after finding the requested record. This
    // validates the complete manifest and detects any second match.
    std::optional<Candidate> selected;
    ManifestHeader header;
    ManifestFooter footer;
    auto consume = [&](uint64_t index, const Candidate &candidate,
                       std::string &) {
      bool match = hasId ? candidate.id == candidateIdOption.getValue()
                         : index == static_cast<uint64_t>(candidateIndex);
      if (match) {
        if (selected)
          return false;
        selected = candidate;
      }
      return true;
    };
    if (!readCandidateManifest(
            candidateFile.getValue(), *taskFacts, func.getSymName(),
            ::mlir::neura::getArchitecture(), /*expectedManifestSha256=*/{},
            /*expectedArchitectureSha256=*/{}, consume, header, footer,
            error)) {
      if (error.empty())
        error = "candidate selection is ambiguous";
      func.emitError() << error;
      return signalPassFailure();
    }
    if (!selected) {
      func.emitError() << "requested candidate is absent from the complete "
                          "manifest";
      return signalPassFailure();
    }

    // Delays mutation until the manifest, IR, architecture, and record have
    // all been validated. These attributes configure the unchanged downstream
    // heuristic mapper; this pass fabricates no placement or II.
    OpBuilder builder(func.getContext());
    for (auto [task, choice] : llvm::zip(*taskFacts, selected->choices)) {
      taskflow::setTaskResourceShape(task.op,
                                     static_cast<int>(choice.shape.cgraCount()),
                                     choice.shape.toCgraShapeAttrValue());
      // The ML model scored this orientation, not merely this rectangle's
      // area. The scheduler may still choose the origin, but it must not turn a
      // scored 1x4 shape into the separately scored 4x1 shape.
      task.op->setAttr("amoeba.analytical_shape_orientation_fixed",
                       builder.getUnitAttr());
    }
    func->setAttr("analytical_task_candidate_id",
                  builder.getStringAttr(selected->id));
    func->setAttr("analytical_task_candidate_scope",
                  builder.getStringAttr(kSearchScope));
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createMaterializeAnalyticalTaskCandidatePass() {
  return std::make_unique<MaterializeAnalyticalTaskCandidatePass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
