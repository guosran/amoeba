//===- MaterializeAnalyticalTaskCandidatePass.cpp ------------------------===//
//
// Implements replay of one validated shape candidate onto Taskflow IR.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

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

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (candidateFile.getValue().empty() ||
        candidateIdOption.getValue().empty()) {
      func.emitError() << "candidates and candidate-id are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    // Continues through the footer after finding the requested record. The
    // reader also rechecks task-body and architecture hashes against this IR
    // and the selected YAML, so a shape cannot be replayed from a stale source
    // task or an incompatible machine specification.
    std::optional<Candidate> selected;
    ManifestHeader header;
    ManifestFooter footer;
    auto consume = [&](uint64_t, const Candidate &candidate, std::string &) {
      if (candidate.id == candidateIdOption.getValue()) {
        if (selected) {
          return false;
        }
        selected = candidate;
      }
      return true;
    };
    if (!readCandidateManifest(
            candidateFile.getValue(), *taskMetadata, func.getSymName(),
            ::mlir::neura::getArchitecture(), consume, header, footer, error)) {
      if (error.empty()) {
        error = "candidate selection is ambiguous";
      }
      func.emitError() << error;
      return signalPassFailure();
    }
    if (!selected) {
      func.emitError() << "requested candidate is absent from the complete "
                          "manifest";
      return signalPassFailure();
    }
    auto graphVariantId =
        func->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    auto graphVariantSha =
        func->getAttrOfType<StringAttr>("amoeba.graph_variant_sha256");
    if ((graphVariantId &&
         graphVariantId.getValue() != header.graphVariantId) ||
        (graphVariantSha &&
         graphVariantSha.getValue() != header.graphVariantSha256) ||
        static_cast<bool>(graphVariantId) != !header.graphVariantId.empty() ||
        static_cast<bool>(graphVariantSha) !=
            !header.graphVariantSha256.empty()) {
      func.emitError() << "candidate manifest graph variant identity does not "
                          "match the selected IR";
      return signalPassFailure();
    }

    // Delays mutation until the manifest, IR, architecture, and record have
    // all been validated. These attributes configure the unchanged downstream
    // heuristic mapper; this pass fabricates no placement or II.
    OpBuilder builder(func.getContext());
    for (auto [task, choice] : llvm::zip(*taskMetadata, selected->choices)) {
      // These attributes are derived from a previous shape, placement, or
      // profiling run. Keeping any of them would let a newly selected
      // rectangle inherit facts for a different candidate.
      for (StringRef attribute :
           {"compiled_ii", "profile_info", "task_orchestration_info",
            "replicas", "tiling", "est_latency"})
        task.op->removeAttr(attribute);
      task.op->setAttr("cgra_count",
                       builder.getI32IntegerAttr(
                           static_cast<int32_t>(choice.shape.cgraCount())));
      task.op->setAttr("cgra_shape", builder.getStringAttr(
                                         choice.shape.toCgraShapeAttrValue()));
      task.op->setAttr("amoeba.selected_cgra_count",
                       builder.getI64IntegerAttr(choice.shape.cgraCount()));
      task.op->setAttr(
          "amoeba.selected_cgra_shape",
          builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
      task.op->setAttr("amoeba.selected_mapper_tile_rows",
                       builder.getI64IntegerAttr(choice.shape.mapperRows));
      task.op->setAttr("amoeba.selected_mapper_tile_cols",
                       builder.getI64IntegerAttr(choice.shape.mapperCols));
      task.op->setAttr("amoeba.selected_trip_count",
                       builder.getI64IntegerAttr(choice.tripCount));
      // The rectangle orientation is part of the selected candidate and must
      // remain unchanged during resource allocation.
      task.op->setAttr("amoeba.joint_shape_orientation_fixed",
                       builder.getUnitAttr());
    }
    func->setAttr("joint_scheduling_candidate_id",
                  builder.getStringAttr(selected->id));
    func->setAttr("joint_scheduling_candidate_scope",
                  builder.getStringAttr(kSearchScope));
    func->setAttr("joint_scheduling_architecture_sha256",
                  builder.getStringAttr(header.architectureSha256));
    if (!header.graphVariantId.empty()) {
      func->setAttr("joint_scheduling_graph_variant_id",
                    builder.getStringAttr(header.graphVariantId));
      func->setAttr("joint_scheduling_graph_variant_sha256",
                    builder.getStringAttr(header.graphVariantSha256));
    }
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
