//===- JointSchedulingRuntimeContractTest.cpp ----------------------------===//
//
// Focused executable tests for typed edges and communication reservations.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"

#include "TaskflowDialect/TaskflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

bool expect(bool condition, StringRef message) {
  if (!condition)
    llvm::errs() << "FAIL: " << message << "\n";
  return condition;
}

FailureOr<InterTaskNetworkSpec> parseNetwork(StringRef yaml,
                                             std::string &error) {
  FailureOr<std::optional<InterTaskNetworkSpec>> parsed =
      parseInterTaskNetworkYaml(yaml, true, error);
  if (failed(parsed) || !parsed->has_value())
    return failure();
  return std::move(**parsed);
}

bool testBitContractAndBottleneckOnce() {
  constexpr StringLiteral yaml = R"yaml(
inter_task_network:
  version: 1
  rows: 1
  columns: 3
  local_bandwidth_bits_per_cycle: 16
  links:
    - src_row: 0
      src_column: 0
      dst_row: 0
      dst_column: 1
      latency_cycles: 2
      bandwidth_bits_per_cycle: 8
    - src_row: 0
      src_column: 1
      dst_row: 0
      dst_column: 2
      latency_cycles: 3
      bandwidth_bits_per_cycle: 4
)yaml";
  std::string error;
  FailureOr<InterTaskNetworkSpec> network = parseNetwork(yaml, error);
  if (!expect(succeeded(network), "heterogeneous network parses"))
    return false;
  FailureOr<int64_t> cycles =
      network->transferCycles({0, 0}, {0, 2}, 33, error);
  if (!expect(succeeded(cycles) && *cycles == 14,
              "two-hop transfer charges bottleneck serialization once"))
    return false;
  cycles = network->transferCycles({0, 1}, {0, 1}, 33, error);
  if (!expect(succeeded(cycles) && *cycles == 3,
              "local payload uses checked bit-level ceil"))
    return false;
  InterTaskNetworkSpec overflow(InterTaskNetworkSpec::kSchemaVersion, 1, 1, 1,
                                {});
  cycles = overflow.transferCycles({0, 0}, {0, 0}, UINT64_MAX, error);
  return expect(failed(cycles), "signed-cycle overflow fails closed");
}

OwningOpRef<ModuleOp> parseTrafficModule(MLIRContext &context) {
  constexpr StringLiteral source = R"mlir(
module {
  func.func @traffic() {
    %flag = taskflow.task @producer : () -> (i1) {
      %true = arith.constant true
      taskflow.yield values(%true : i1)
    }
    %channel = taskflow.channel %flag : i1 -> i1
    taskflow.task @consumer_a value_inputs(%channel : i1) : (i1) -> () {
    ^bb0(%value: i1):
      taskflow.yield
    }
    taskflow.task @consumer_b value_inputs(%flag : i1)
        {amoeba.control_predecessors = ["producer"]} : (i1) -> () {
    ^bb0(%value: i1):
      taskflow.yield
    }
    return
  }
}
)mlir";
  return parseSourceString<ModuleOp>(source, &context);
}

bool testTypedEdgesAndReservations() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, TaskflowDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module = parseTrafficModule(context);
  if (!expect(static_cast<bool>(module), "typed-edge module parses"))
    return false;
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("traffic");
  std::string error;
  TaskEdgeGraphOptions options;
  options.require_payload = true;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(function, options, error);
  if (!expect(succeeded(graph), "strict typed graph builds")) {
    llvm::errs() << error << "\n";
    return false;
  }
  unsigned valueEdges = 0;
  unsigned controlEdges = 0;
  for (const TaskEdge &edge : graph->getEdges()) {
    if (edge.kind == TaskEdgeKind::Value) {
      ++valueEdges;
      if (!expect(edge.payload_bits && *edge.payload_bits == 1,
                  "i1 payload remains one bit"))
        return false;
    }
    if (edge.kind == TaskEdgeKind::Control)
      ++controlEdges;
  }
  if (!expect(valueEdges == 2 && controlEdges == 1,
              "channel, SSA, and control edges remain distinct"))
    return false;

  TaskflowTaskOp producer;
  TaskflowTaskOp consumerA;
  TaskflowTaskOp consumerB;
  function.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == "producer")
      producer = task;
    else if (task.getTaskName() == "consumer_a")
      consumerA = task;
    else if (task.getTaskName() == "consumer_b")
      consumerB = task;
  });

  constexpr StringLiteral routedYaml = R"yaml(
inter_task_network:
  version: 1
  rows: 2
  columns: 2
  local_bandwidth_bits_per_cycle: 1
  links:
    - {src_row: 0, src_column: 0, dst_row: 0, dst_column: 1, latency_cycles: 1, bandwidth_bits_per_cycle: 1}
    - {src_row: 0, src_column: 0, dst_row: 1, dst_column: 0, latency_cycles: 1, bandwidth_bits_per_cycle: 1}
    - {src_row: 1, src_column: 0, dst_row: 0, dst_column: 1, latency_cycles: 1, bandwidth_bits_per_cycle: 1}
)yaml";
  FailureOr<InterTaskNetworkSpec> network = parseNetwork(routedYaml, error);
  if (!expect(succeeded(network), "competing-route network parses"))
    return false;
  InterTaskNetworkCommunicationModel routed(*graph, *network);
  routed.beginTrial();
  int64_t readyA = 0;
  int64_t readyB = 0;
  if (!expect(routed.getTransferReadyCycle(producer, consumerA, 0, 0, 0, 1, 0,
                                           true, readyA, error) &&
                  readyA == 2,
              "first transfer reserves the shortest route") ||
      !expect(routed.getTransferReadyCycle(producer, consumerB, 0, 0, 0, 1, 0,
                                           true, readyB, error) &&
                  readyB == 3,
              "second transfer uses load-aware alternate route"))
    return false;
  routed.finishTrial(true);
  if (!expect(routed.getCommittedTransfers().size() == 2 &&
                  routed.getCommittedTransfers()[1].links.size() == 2,
              "shared-link competition is preserved in the trace"))
    return false;
  const auto &alternate = routed.getCommittedTransfers()[1];
  if (!expect(alternate.pathLatencyCycles == 2 &&
                  alternate.bottleneckBandwidthBitsPerCycle == 1 &&
                  alternate.transferCycles == 3,
              "route trace records latency, bottleneck, and transfer cycles"))
    return false;

  InterTaskNetworkCommunicationModel local(*graph, *network);
  local.beginTrial();
  readyA = 0;
  readyB = 0;
  if (!expect(local.getTransferReadyCycle(producer, consumerA, 0, 0, 0, 0, 7,
                                          true, readyA, error) &&
                  readyA == 8,
              "first local transfer reserves its coordinate channel") ||
      !expect(local.getTransferReadyCycle(producer, consumerB, 0, 0, 0, 0, 7,
                                          true, readyB, error) &&
                  readyB == 9,
              "overlapping local transfer waits for channel capacity"))
    return false;
  local.finishTrial(true);
  if (!expect(local.getCommittedTransfers()[1].links.front().localChannel,
              "local reservation has an explicit trace identity"))
    return false;

  SmallVector<InterTaskNetworkLink> denseLinks;
  for (int source = 0; source < 16; ++source)
    for (int destination = 0; destination < 16; ++destination)
      if (source != destination)
        denseLinks.push_back({{source / 4, source % 4},
                              {destination / 4, destination % 4},
                              1,
                              1});
  InterTaskNetworkSpec dense(InterTaskNetworkSpec::kSchemaVersion, 4, 4, 1,
                             std::move(denseLinks));
  if (!expect(dense.validateForGrid(4, 4, error),
              "dense cyclic network validates"))
    return false;
  InterTaskNetworkCommunicationModel bounded(*graph, std::move(dense));
  bounded.beginTrial();
  int64_t ignored = 0;
  return expect(!bounded.getTransferReadyCycle(producer, consumerA, 0, 0, 3, 3,
                                               0, false, ignored, error) &&
                    error.find("deterministic budget") != std::string::npos,
                "dense cyclic route search terminates by failing closed");
}

bool testDynamicPayloadFailsClosed() {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, TaskflowDialect>();
  MLIRContext context(registry);
  constexpr StringLiteral source = R"mlir(
module {
  func.func @dynamic(%input: memref<?xi1>) {
    %state = taskflow.task @producer will_writes(%input : memref<?xi1>)
        [original_write_memrefs(%input : memref<?xi1>)]
        : (memref<?xi1>) -> (memref<?xi1>) {
    ^bb0(%buffer: memref<?xi1>):
      taskflow.yield done_writes(%buffer : memref<?xi1>)
    }
    taskflow.task @consumer will_reads(%state : memref<?xi1>)
        [original_read_memrefs(%input : memref<?xi1>)]
        : (memref<?xi1>) -> () {
    ^bb0(%buffer: memref<?xi1>):
      taskflow.yield
    }
    return
  }
}
)mlir";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!expect(static_cast<bool>(module), "dynamic-payload module parses"))
    return false;
  TaskEdgeGraphOptions options;
  options.require_payload = true;
  std::string error;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
      module->lookupSymbol<func::FuncOp>("dynamic"), options, error);
  return expect(failed(graph) &&
                    error.find("unknown payload") != std::string::npos,
                "dynamic payload without an explicit bit count fails closed");
}

bool testJoinExpandsTypedRegionEdges() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, TaskflowDialect>();
  MLIRContext context(registry);
  constexpr StringLiteral source = R"mlir(
module {
  func.func @joined(%output: memref<4x2xi8>) {
    %left = taskflow.task @left will_writes(%output : memref<4x2xi8>)
        [original_write_memrefs(%output : memref<4x2xi8>)]
        : (memref<4x2xi8>) -> (memref<4x2xi8>) {
    ^bb0(%write: memref<4x2xi8>):
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c2 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c2 step %c1 : index
      taskflow.yield done_writes(%write : memref<4x2xi8>)
    }
    %right = taskflow.task @right will_writes(%output : memref<4x2xi8>)
        [original_write_memrefs(%output : memref<4x2xi8>)]
        : (memref<4x2xi8>) -> (memref<4x2xi8>) {
    ^bb0(%write: memref<4x2xi8>):
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c2 to %c4 step %c1 : index
      %j = taskflow.counter parent(%i : index) from %c0 to %c2 step %c1 : index
      taskflow.yield done_writes(%write : memref<4x2xi8>)
    }
    %joined = taskflow.join states(%left, %right) base(%output) axis(0)
        region([0, 0], [4, 2]) : memref<4x2xi8>
    taskflow.task @consumer will_reads(%joined : memref<4x2xi8>)
        [original_read_memrefs(%output : memref<4x2xi8>)]
        : (memref<4x2xi8>) -> () {
    ^bb0(%read: memref<4x2xi8>):
      taskflow.yield
    }
    return
  }
}
)mlir";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!expect(static_cast<bool>(module), "joined-region module parses"))
    return false;
  std::string error;
  TaskEdgeGraphOptions options;
  options.require_payload = true;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
      module->lookupSymbol<func::FuncOp>("joined"), options, error);
  if (!expect(succeeded(graph), "taskflow.join expands to typed producers")) {
    llvm::errs() << error << "\n";
    return false;
  }
  unsigned rawEdges = 0;
  for (const TaskEdge &edge : graph->getEdges()) {
    if (edge.kind != TaskEdgeKind::Raw)
      continue;
    ++rawEdges;
    if (!expect(edge.payload_bits && *edge.payload_bits == 32,
                "each tile edge carries its exact counter-domain region"))
      return false;
  }
  return expect(rawEdges == 2,
                "joined consumer depends on every disjoint tile producer");
}

} // namespace

int main() {
  bool success = true;
  success &= testBitContractAndBottleneckOnce();
  success &= testTypedEdgesAndReservations();
  success &= testDynamicPayloadFailsClosed();
  success &= testJoinExpandsTypedRegionEdges();
  if (success)
    llvm::outs() << "Joint scheduling runtime contract tests passed\n";
  return success ? 0 : 1;
}
