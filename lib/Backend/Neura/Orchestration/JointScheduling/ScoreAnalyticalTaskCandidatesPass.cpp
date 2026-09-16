//===- ScoreAnalyticalTaskCandidatesPass.cpp -----------------------------===//
//
// Implements full-manifest scoring and deterministic top-k selection.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"
#include "AnalyticalTaskCostCatalog.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

static std::string hashBytes(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

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
           "cost cache, then emits deterministic top-k";
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
  Option<std::string> communicationMode{
      *this, "communication-mode",
      llvm::cl::desc("Task communication mode: 'none' or 'explicit'."),
      llvm::cl::init("none")};
  Option<std::string> graphVariantId{
      *this, "graph-variant-id",
      llvm::cl::desc("Optional graph variant identity; requires its SHA-256."),
      llvm::cl::init("")};
  Option<std::string> graphVariantSha256{
      *this, "graph-variant-sha256",
      llvm::cl::desc("SHA-256 of the exact graph-variant result IR bytes."),
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
    if (graphVariantId.getValue().empty() !=
        graphVariantSha256.getValue().empty()) {
      func.emitError() << "graph-variant-id and graph-variant-sha256 must be "
                          "provided together";
      return signalPassFailure();
    }
    if (candidateFile.getValue().empty() || costFile.getValue().empty() ||
        outputFile.getValue().empty() || topK.getValue() < 0 ||
        samePath(candidateFile.getValue(), outputFile.getValue()) ||
        samePath(costFile.getValue(), outputFile.getValue()) ||
        samePath(candidateFile.getValue(), costFile.getValue())) {
      func.emitError() << "distinct candidates, cost-file, and output paths "
                          "and non-negative top-k are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    // Bind scoring to the exact JSONL bytes that enumerate the candidate
    // space.  This includes the manifest header, candidate ordering, and
    // footer; a same-shaped space serialized differently is still a different
    // input to the cost catalogue.
    FailureOr<std::string> candidateSha =
        sha256File(candidateFile.getValue(), error);
    // Bind scoring to the exact architecture YAML selected for this pass.
    // The manifest and catalogue use this identity to reject costs generated
    // for a same-sized but otherwise different machine.
    FailureOr<std::string> architectureSha = currentArchitectureSha256(error);
    if (failed(candidateSha) || failed(architectureSha)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    if (!costs.load(costFile.getValue(), func.getSymName(), *taskMetadata,
                    *candidateSha, *architectureSha, error)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    if (communicationMode.getValue() != "none" &&
        communicationMode.getValue() != "explicit") {
      func.emitError() << "unknown communication mode: "
                       << communicationMode.getValue();
      return signalPassFailure();
    }
    const bool communicationAware = communicationMode.getValue() == "explicit";
    FailureOr<std::optional<InterTaskNetworkSpec>> network =
        loadInterTaskNetworkSpec(::mlir::amoeba::getNeuraArchitectureSpecFile(),
                                 /*require_network=*/communicationAware, error);
    if (failed(network)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Scores are held until readCandidateManifest has independently proven
    // canonical ordering, exact packing, and a complete footer.
    SmallVector<RankedCandidate> ranked;
    ManifestHeader manifestHeader;
    ManifestFooter manifestFooter;
    uint64_t scoredCount = 0;
    uint64_t validCount = 0;
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          llvm::json::Object scoreHeader;
          scoreHeader["record_type"] = "header";
          scoreHeader["schema"] = kScoreSchema.str();
          scoreHeader["candidate_schema"] = kCandidateSchema.str();
          scoreHeader["function"] = func.getSymName().str();
          scoreHeader["cost_namespace"] = costs.nameSpace().str();
          // The Python driver checks these hashes before replaying any
          // materialized candidate.  They form a chain from the score output
          // back to the exact manifest, cost JSON, and architecture inputs.
          scoreHeader["candidate_manifest_sha256"] =
              costs.candidateManifestSha256().str();
          scoreHeader["cost_catalog_sha256"] = costs.catalogSha256().str();
          scoreHeader["architecture_sha256"] = costs.architectureSha256().str();
          scoreHeader["score_model"] = kScoreModel.str();
          scoreHeader["communication_mode"] = communicationMode.getValue();
          if (!graphVariantId.getValue().empty()) {
            scoreHeader["graph_variant_id"] = graphVariantId.getValue();
            scoreHeader["graph_variant_sha256"] = graphVariantSha256.getValue();
          }
          scoreHeader["mapper_success_probability"] =
              costs.mapperSuccessProbabilityRole().str();
          writeJsonLine(os, std::move(scoreHeader));

          auto consume = [&](uint64_t manifestIndex, const Candidate &candidate,
                             std::string &consumeError) {
            // duration(task, shape) = startup + II * (trip_count - 1)
            // score(candidate)      = production scheduler makespan
            bool valid = true;
            double bottleneck = 0.0;
            SmallVector<int64_t> schedulerDurations;
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
              taskCost["trip_count"] = choice.tripCount;
              if (!cost->supported) {
                valid = false;
                if (rejectReason.empty())
                  rejectReason = "UNSUPPORTED_TASK_SHAPE";
                taskCost["support_status"] = "unsupported";
              } else {
                taskCost["predicted_ii"] = cost->predictedII;
                taskCost["startup_cycles"] = cost->startupCycles;
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
                double scheduledDuration = std::ceil(duration);
                if (scheduledDuration >
                    static_cast<double>(std::numeric_limits<int64_t>::max())) {
                  consumeError =
                      "task duration exceeds scheduler range for task=" +
                      choice.task;
                  return false;
                }
                schedulerDurations.push_back(std::max<int64_t>(
                    1, static_cast<int64_t>(scheduledDuration)));
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
              // The clone receives the same fixed shapes and task priority as
              // final orchestration. Only its predicted durations differ.
              OwningOpRef<ModuleOp> clonedModule = module.clone();
              FailureOr<func::FuncOp> clonedFunction = selectTaskFunction(
                  *clonedModule, func.getSymName(), consumeError);
              if (failed(clonedFunction))
                return false;
              SmallVector<taskflow::TaskflowTaskOp> clonedTasks;
              clonedFunction->walk([&](taskflow::TaskflowTaskOp task) {
                clonedTasks.push_back(task);
              });
              if (clonedTasks.size() != candidate.choices.size() ||
                  clonedTasks.size() != schedulerDurations.size()) {
                consumeError = "candidate task count changed while cloning";
                return false;
              }
              OpBuilder builder(func.getContext());
              for (auto [task, choice, duration] : llvm::zip_equal(
                       clonedTasks, candidate.choices, schedulerDurations)) {
                task->setAttr("cgra_count", builder.getI32IntegerAttr(
                                                choice.shape.cgraCount()));
                task->setAttr(
                    "cgra_shape",
                    builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
                task->setAttr("amoeba.joint_shape_orientation_fixed",
                              builder.getUnitAttr());
                task->setAttr("est_latency",
                              builder.getI64IntegerAttr(duration));
              }
              taskflow::AnalyticalBasedTaskOrchestration strategy(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  taskflow::SchedulingMode::SpatialTemporal);
              std::unique_ptr<InterTaskNetworkCommunicationModel>
                  communicationModel;
              if (communicationAware) {
                FailureOr<TaskEdgeGraph> edgeGraph = buildTaskEdgeGraph(
                    *clonedFunction,
                    TaskEdgeGraphOptions{/*require_payload=*/true,
                                         /*read_control_predecessors=*/true},
                    consumeError);
                if (failed(edgeGraph))
                  return false;
                communicationModel =
                    std::make_unique<InterTaskNetworkCommunicationModel>(
                        std::move(*edgeGraph), **network);
              }
              taskflow::TaskScheduler scheduler(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  taskflow::SchedulingMode::SpatialTemporal,
                  taskflow::ShapeSelectionPolicy::FixedOrientation,
                  communicationModel.get());
              if (!scheduler.schedule(
                      *clonedFunction,
                      strategy.computeRoutingCriticalPathPriority(
                          *clonedFunction))) {
                consumeError =
                    "spatial-temporal scheduler rejected candidate " +
                    candidate.id;
                return false;
              }
              int64_t makespan = scheduler.getScheduleMakespan();
              if (makespan <= 0) {
                consumeError = "scheduler produced an empty or overflowing "
                               "makespan for " +
                               candidate.id;
                return false;
              }
              score["predicted_scheduler_makespan"] = makespan;
              llvm::json::Array taskSchedule;
              std::string traceIdentity;
              llvm::raw_string_ostream identity(traceIdentity);
              identity << "joint-schedule-trace-v2\n"
                       << candidate.id << "\n"
                       << *architectureSha << "\n"
                       << communicationMode.getValue() << "\n";
              for (taskflow::TaskflowTaskOp task : clonedTasks) {
                auto scheduleEntry = llvm::find_if(
                    scheduler.getScheduleEntries(), [&](const auto &entry) {
                      return entry.task == task.getOperation();
                    });
                if (scheduleEntry == scheduler.getScheduleEntries().end()) {
                  consumeError = "scheduler did not emit a complete task trace";
                  return false;
                }
                llvm::json::Object taskRecord;
                taskRecord["task"] = task.getTaskName().str();
                taskRecord["start_cycle"] = scheduleEntry->startCycle;
                taskRecord["end_cycle"] = scheduleEntry->endCycle;
                identity << "T|" << task.getTaskName().size() << ":"
                         << task.getTaskName() << "|"
                         << scheduleEntry->startCycle << "|"
                         << scheduleEntry->endCycle;
                llvm::json::Array cells;
                for (auto [row, col] : scheduleEntry->positions) {
                  llvm::json::Object cell;
                  cell["row"] = row;
                  cell["col"] = col;
                  identity << "|" << row << "," << col;
                  cells.push_back(std::move(cell));
                }
                identity << "\n";
                taskRecord["cgra_positions"] = std::move(cells);
                taskSchedule.push_back(std::move(taskRecord));
              }
              llvm::json::Array routes;
              llvm::json::Array dependencies;
              if (communicationModel) {
                for (const TaskEdge &edge :
                     communicationModel->getTypedEdges()) {
                  taskflow::TaskflowTaskOp producer = edge.producer;
                  taskflow::TaskflowTaskOp consumer = edge.consumer;
                  llvm::json::Object dependency;
                  dependency["producer"] = producer.getTaskName().str();
                  dependency["consumer"] = consumer.getTaskName().str();
                  dependency["kind"] = stringifyTaskEdgeKind(edge.kind).str();
                  dependency["producer_segment"] =
                      stringifyTaskResultSegment(edge.producer_segment).str();
                  dependency["producer_index"] = edge.producer_index;
                  dependency["consumer_segment"] =
                      stringifyTaskOperandSegment(edge.consumer_segment).str();
                  dependency["consumer_index"] = edge.consumer_index;
                  if (edge.payload_bits)
                    dependency["payload_bits"] =
                        static_cast<int64_t>(*edge.payload_bits);
                  identity << "D|" << producer.getTaskName().size() << ":"
                           << producer.getTaskName() << "|"
                           << consumer.getTaskName().size() << ":"
                           << consumer.getTaskName() << "|"
                           << stringifyTaskEdgeKind(edge.kind) << "|"
                           << stringifyTaskResultSegment(edge.producer_segment)
                           << "|" << edge.producer_index << "|"
                           << stringifyTaskOperandSegment(edge.consumer_segment)
                           << "|" << edge.consumer_index << "|";
                  if (edge.payload_bits)
                    identity << *edge.payload_bits;
                  else
                    identity << "-";
                  identity << "\n";
                  dependencies.push_back(std::move(dependency));
                }
                for (const auto &transfer :
                     communicationModel->getCommittedTransfers()) {
                  llvm::json::Object route;
                  route["producer"] =
                      cast<taskflow::TaskflowTaskOp>(transfer.producer)
                          .getTaskName()
                          .str();
                  route["consumer"] =
                      cast<taskflow::TaskflowTaskOp>(transfer.consumer)
                          .getTaskName()
                          .str();
                  if (transfer.payloadBits >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.pathLatencyCycles >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.bottleneckBandwidthBitsPerCycle >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.transferCycles >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max())) {
                    consumeError =
                        "communication trace value exceeds signed 64-bit";
                    return false;
                  }
                  route["payload_bits"] =
                      static_cast<int64_t>(transfer.payloadBits);
                  route["path_latency_cycles"] =
                      static_cast<int64_t>(transfer.pathLatencyCycles);
                  route["bottleneck_bandwidth_bits_per_cycle"] =
                      static_cast<int64_t>(
                          transfer.bottleneckBandwidthBitsPerCycle);
                  route["transfer_cycles"] =
                      static_cast<int64_t>(transfer.transferCycles);
                  route["ready_cycle"] = transfer.readyCycle;
                  route["source_row"] = transfer.source.row;
                  route["source_col"] = transfer.source.column;
                  route["destination_row"] = transfer.destination.row;
                  route["destination_col"] = transfer.destination.column;
                  StringRef producerName =
                      cast<taskflow::TaskflowTaskOp>(transfer.producer)
                          .getTaskName();
                  StringRef consumerName =
                      cast<taskflow::TaskflowTaskOp>(transfer.consumer)
                          .getTaskName();
                  identity << "R|" << producerName.size() << ":" << producerName
                           << "|" << consumerName.size() << ":" << consumerName
                           << "|" << transfer.source.row << ","
                           << transfer.source.column << "|"
                           << transfer.destination.row << ","
                           << transfer.destination.column << "|"
                           << transfer.payloadBits << "|" << transfer.readyCycle
                           << "|" << transfer.pathLatencyCycles << "|"
                           << transfer.bottleneckBandwidthBitsPerCycle << "|"
                           << transfer.transferCycles;
                  llvm::json::Array edgeIndices;
                  for (uint32_t edgeIndex : transfer.edgeIndices) {
                    edgeIndices.push_back(edgeIndex);
                    identity << "|e" << edgeIndex;
                  }
                  route["edge_indices"] = std::move(edgeIndices);
                  llvm::json::Array links;
                  for (const auto &interval : transfer.links) {
                    llvm::json::Object link;
                    link["start_cycle"] = interval.startCycle;
                    link["end_cycle"] = interval.endCycle;
                    if (interval.localChannel) {
                      link["resource_kind"] = "local_channel";
                      link["row"] = interval.localCoordinate.row;
                      link["col"] = interval.localCoordinate.column;
                      identity << "|c" << interval.localCoordinate.row << ","
                               << interval.localCoordinate.column << ","
                               << interval.startCycle << ","
                               << interval.endCycle;
                    } else {
                      link["resource_kind"] = "network_link";
                      link["link_index"] = interval.linkIndex;
                      identity << "|l" << interval.linkIndex << ","
                               << interval.startCycle << ","
                               << interval.endCycle;
                    }
                    links.push_back(std::move(link));
                  }
                  identity << "\n";
                  route["links"] = std::move(links);
                  routes.push_back(std::move(route));
                }
              }
              llvm::json::Object trace;
              trace["candidate_id"] = candidate.id;
              trace["architecture_sha256"] = *architectureSha;
              trace["communication_mode"] = communicationMode.getValue();
              trace["task_schedule"] = std::move(taskSchedule);
              trace["dependencies"] = std::move(dependencies);
              trace["routes"] = std::move(routes);
              llvm::json::Value traceValue(std::move(trace));
              identity.flush();
              score["schedule_trace_sha256"] = hashBytes(traceIdentity);
              score["schedule_trace"] = std::move(traceValue);
              ranked.push_back(
                  {candidate.id, manifestIndex, static_cast<double>(makespan)});
              ++validCount;
            } else {
              score["reject_reason"] = rejectReason;
            }
            writeJsonLine(os, std::move(score));
            ++scoredCount;
            return true;
          };

          if (!readCandidateManifest(candidateFile.getValue(), *taskMetadata,
                                     func.getSymName(),
                                     ::mlir::neura::getArchitecture(), consume,
                                     manifestHeader, manifestFooter, error))
            return false;
          if (manifestHeader.architectureSha256 != costs.architectureSha256() ||
              manifestHeader.graphVariantId != graphVariantId.getValue() ||
              manifestHeader.graphVariantSha256 !=
                  graphVariantSha256.getValue() ||
              scoredCount != manifestFooter.candidateCount) {
            error = "not every frozen candidate was scored under the bound "
                    "architecture";
            return false;
          }
          if (costs.coveredQueries() != costs.catalogQueries()) {
            error = "cost catalogue does not exactly cover candidate manifest "
                    "task-shape queries";
            return false;
          }

          // Sort only after complete traversal. Numeric manifest order is the
          // deterministic tie-break, avoiding candidate-13 < candidate-5.
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
            item["predicted_scheduler_makespan"] = ranked[index].score;
            shortlist.push_back(std::move(item));
          }

          llvm::json::Object cacheStats;
          cacheStats["hits"] = static_cast<int64_t>(costs.hits());
          cacheStats["misses"] = static_cast<int64_t>(costs.misses());
          cacheStats["predictions"] =
              static_cast<int64_t>(costs.cachedPredictions());
          cacheStats["queries"] = static_cast<int64_t>(costs.coveredQueries());
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
    llvm::errs() << "[JointScheduling] scored all " << scoredCount
                 << " candidates with " << costs.cachedPredictions()
                 << " cached task-body/shape predictions\n";
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
