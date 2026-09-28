//===- EnumerateJointSemanticRewritePass.cpp -----------------*- C++ -*-===//
//
// Hardware independent semantic rewrite coordinator for the canonical
// producer/consumer GEMM pipeline.  This pass only reads Taskflow IR and
// writes a semantic candidate manifest.  Individual rewrite passes consume
// the typed action records; this pass never edits the source module.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
using namespace mlir::taskflow;

namespace {

constexpr int64_t kDimension = 8;
constexpr llvm::StringLiteral kGraphSchema = "orbit-joint-semantic-graph-v1";
constexpr llvm::StringLiteral kDagSchema = "orbit-joint-rewrite-dag-v1";
constexpr llvm::StringLiteral kActionSchema = "orbit-joint-rewrite-action-v1";
constexpr llvm::StringLiteral kRegionSchema = "orbit-half-open-region-v1";
constexpr llvm::StringLiteral kArithmeticSemantics = "i32_modulo_2^32";

using Range = std::array<int64_t, 2>;

static std::string hashBytes(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static std::optional<int64_t> constantIndex(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValueAttr());
  if (!integer || !integer.getType().isIndex())
    return std::nullopt;
  return integer.getInt();
}

static bool isEightByEightI32(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && type.getShape()[0] == kDimension &&
         type.getShape()[1] == kDimension &&
         type.getElementType().isInteger(32);
}

static bool isEightByEightI32Type(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 2 &&
         memref.getShape()[0] == kDimension &&
         memref.getShape()[1] == kDimension &&
         memref.getElementType().isInteger(32);
}

static bool hasNoAliasAttr(func::FuncOp function, unsigned index) {
  return static_cast<bool>(function.getArgAttr(index, "amoeba.noalias"));
}

static bool equalValues(ValueRange actual, ArrayRef<Value> expected) {
  return actual.size() == expected.size() && llvm::equal(actual, expected);
}

static bool containsValue(ValueRange values, Value value) {
  return llvm::is_contained(values, value);
}

static bool isKnownCanonicalBodyOperation(Operation *operation) {
  return isa<arith::AddIOp, arith::ConstantOp, arith::MulIOp, memref::LoadOp,
             memref::StoreOp, scf::ForOp, scf::YieldOp, TaskflowCounterOp,
             TaskflowHyperblockOp, TaskflowHyperblockYieldOp, TaskflowYieldOp>(
      operation);
}

static bool checkKnownBodyOperations(TaskflowTaskOp task, std::string &error) {
  bool valid = true;
  task.walk([&](Operation *operation) {
    if (operation == task.getOperation())
      return WalkResult::advance();
    if (isKnownCanonicalBodyOperation(operation))
      return WalkResult::advance();
    error = "UNKNOWN_ACCESS: task " + task.getTaskName().str() +
            " contains unsupported operation " +
            operation->getName().getStringRef().str();
    valid = false;
    return WalkResult::interrupt();
  });
  return valid;
}

static bool checkStaticCounterChain(TaskflowTaskOp task,
                                    TaskflowHyperblockOp hyperblock,
                                    unsigned expectedCounters,
                                    std::string &error) {
  SmallVector<TaskflowCounterOp> counters;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
  if (counters.size() != expectedCounters) {
    error = "DYNAMIC_CONTROL: task " + task.getTaskName().str() +
            " must contain exactly two canonical counters";
    return false;
  }
  if (hyperblock.getIndices().size() != counters.size()) {
    error = "DYNAMIC_CONTROL: hyperblock counter arity does not match the "
            "canonical loop nest";
    return false;
  }
  for (auto indexed : llvm::enumerate(counters)) {
    TaskflowCounterOp counter = indexed.value();
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *lower != 0 || *upper != kDimension ||
        *step != 1) {
      error = "DYNAMIC_CONTROL: canonical counters require static 0..8 step 1 "
              "domains";
      return false;
    }
    if (indexed.index() == 0) {
      if (counter.getParentIndex()) {
        error = "DYNAMIC_CONTROL: the outer canonical counter cannot have a "
                "parent";
        return false;
      }
    } else if (!counter.getParentIndex() ||
               counter.getParentIndex() !=
                   counters[indexed.index() - 1].getCounterIndex()) {
      error = "DYNAMIC_CONTROL: canonical counters must form one parent "
              "chain";
      return false;
    }
    if (hyperblock.getIndices()[indexed.index()] != counter.getCounterIndex()) {
      error = "DYNAMIC_CONTROL: hyperblock indices do not match counters";
      return false;
    }
  }
  if (!hyperblock.getIterArgs().empty() || hyperblock.getNumResults() != 0) {
    error = "DYNAMIC_CONTROL: hyperblock iter_args/results are not supported";
    return false;
  }
  if (hyperblock.getBody().empty() ||
      hyperblock.getBody().front().getNumArguments() != expectedCounters) {
    error = "DYNAMIC_CONTROL: canonical hyperblock must have two index "
            "arguments";
    return false;
  }
  return true;
}

static TaskflowHyperblockOp findSingleHyperblock(TaskflowTaskOp task,
                                                 std::string &error) {
  TaskflowHyperblockOp result;
  task.walk([&](TaskflowHyperblockOp hyperblock) {
    if (result) {
      error = "MULTIPLE_EXITS: task " + task.getTaskName().str() +
              " contains multiple hyperblocks";
      return WalkResult::interrupt();
    }
    result = hyperblock;
    return WalkResult::advance();
  });
  if (!result && error.empty())
    error = "DYNAMIC_CONTROL: task " + task.getTaskName().str() +
            " must contain one hyperblock";
  return result;
}

static scf::ForOp findSingleKLoop(TaskflowTaskOp task, std::string &error) {
  scf::ForOp result;
  task.walk([&](scf::ForOp loop) {
    if (result) {
      error = "DYNAMIC_CONTROL: producer must contain one K loop";
      return WalkResult::interrupt();
    }
    result = loop;
    return WalkResult::advance();
  });
  if (!result && error.empty())
    error = "DYNAMIC_CONTROL: producer must contain one static K loop";
  return result;
}

static bool checkStaticKLoop(scf::ForOp loop, std::string &error) {
  std::optional<int64_t> lower = constantIndex(loop.getLowerBound());
  std::optional<int64_t> upper = constantIndex(loop.getUpperBound());
  std::optional<int64_t> step = constantIndex(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *upper != kDimension ||
      *step != 1 || loop.getNumRegionIterArgs() != 1 ||
      loop.getNumResults() != 1) {
    error = "DYNAMIC_CONTROL: canonical K loop requires static 0..8 step 1 "
            "and one i32 reduction value";
    return false;
  }
  if (loop.getBody()->getNumArguments() != 2 ||
      !isa<scf::YieldOp>(loop.getBody()->getTerminator())) {
    error = "DYNAMIC_CONTROL: canonical K loop body is malformed";
    return false;
  }
  return true;
}

static bool checkTaskYield(TaskflowTaskOp task, unsigned expectedWriteArg,
                           std::string &error) {
  if (task.getBody().empty()) {
    error = "MULTIPLE_EXITS: task body is empty";
    return false;
  }
  auto yield =
      dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
  if (!yield || !yield.getDoneReads().empty() ||
      yield.getDoneWrites().size() != 1 || !yield.getValueResults().empty() ||
      yield.getDoneWrites().front() !=
          task.getBody().front().getArgument(expectedWriteArg)) {
    error = "EDGE_ENDPOINT_INVALID: task yield does not return its canonical "
            "write state";
    return false;
  }
  return true;
}

static bool checkProducerBody(TaskflowTaskOp producer,
                              TaskflowHyperblockOp hyperblock, scf::ForOp kLoop,
                              std::string &error) {
  Block &taskBody = producer.getBody().front();
  Block &hyperblockBody = hyperblock.getBody().front();
  Block &loopBody = *kLoop.getBody();
  BlockArgument readA = taskBody.getArgument(0);
  BlockArgument readB = taskBody.getArgument(1);
  BlockArgument readC = taskBody.getArgument(2);
  BlockArgument writeT = taskBody.getArgument(3);
  Value outerM = hyperblockBody.getArgument(0);
  Value outerN = hyperblockBody.getArgument(1);
  Value innerK = kLoop.getInductionVar();
  Value loopAccumulator = loopBody.getArgument(1);

  memref::LoadOp loadA;
  memref::LoadOp loadB;
  memref::LoadOp loadC;
  memref::StoreOp storeT;
  arith::MulIOp multiply;
  arith::AddIOp add;
  scf::YieldOp loopYield;
  unsigned aLoads = 0;
  unsigned bLoads = 0;
  unsigned cLoads = 0;
  unsigned tStores = 0;
  unsigned multiplyCount = 0;
  unsigned addCount = 0;
  unsigned loopYieldCount = 0;

  producer.walk([&](memref::LoadOp load) {
    if (load.getMemRef() == readA &&
        equalValues(load.getIndices(), {outerM, innerK})) {
      loadA = load;
      ++aLoads;
      return;
    }
    if (load.getMemRef() == readB &&
        equalValues(load.getIndices(), {innerK, outerN})) {
      loadB = load;
      ++bLoads;
      return;
    }
    if (load.getMemRef() == readC &&
        equalValues(load.getIndices(), {outerM, outerN})) {
      loadC = load;
      ++cLoads;
      return;
    }
    error = "UNKNOWN_ACCESS: producer has a load outside the exact GEMM "
            "relations";
  });
  producer.walk([&](memref::StoreOp store) {
    if (store.getMemRef() == writeT &&
        equalValues(store.getIndices(), {outerM, outerN})) {
      storeT = store;
      ++tStores;
      return;
    }
    error = "UNKNOWN_ACCESS: producer has a store outside the exact T "
            "relation";
  });
  producer.walk([&](arith::MulIOp op) {
    multiply = op;
    ++multiplyCount;
  });
  producer.walk([&](arith::AddIOp op) {
    add = op;
    ++addCount;
  });
  producer.walk([&](scf::YieldOp yield) {
    loopYield = yield;
    ++loopYieldCount;
  });
  if (!error.empty())
    return false;
  if (aLoads != 1 || bLoads != 1 || cLoads != 1 || tStores != 1 ||
      multiplyCount != 1 || addCount != 1 || loopYieldCount != 1 || !loadA ||
      !loadB || !loadC || !storeT || !multiply || !add || !loopYield) {
    error = "UNKNOWN_ACCESS: producer does not match the canonical one-MAC "
            "GEMM body";
    return false;
  }
  if (loadA->getBlock() != &loopBody || loadB->getBlock() != &loopBody ||
      loadC->getBlock() != &hyperblockBody ||
      storeT->getBlock() != &hyperblockBody ||
      multiply->getBlock() != &loopBody || add->getBlock() != &loopBody ||
      loopYield->getBlock() != &loopBody) {
    error = "UNKNOWN_ACCESS: producer accesses are outside the canonical "
            "loop regions";
    return false;
  }
  if (!containsValue(multiply.getOperands(), loadA.getResult()) ||
      !containsValue(multiply.getOperands(), loadB.getResult()) ||
      !containsValue(add.getOperands(), multiply.getResult()) ||
      !containsValue(add.getOperands(), loopAccumulator) ||
      kLoop.getInitArgs().size() != 1 ||
      kLoop.getInitArgs().front() != loadC.getResult() ||
      loopYield.getResults().size() != 1 ||
      loopYield.getResults().front() != add.getResult() ||
      storeT.getValue() != kLoop.getResult(0)) {
    error = "EDGE_REGION_INVALID: producer arithmetic does not carry C through "
            "the exact K reduction";
    return false;
  }
  return true;
}

static bool checkConsumerBody(TaskflowTaskOp consumer,
                              TaskflowHyperblockOp hyperblock,
                              std::string &error) {
  Block &taskBody = consumer.getBody().front();
  Block &hyperblockBody = hyperblock.getBody().front();
  BlockArgument readT = taskBody.getArgument(0);
  BlockArgument readE = taskBody.getArgument(1);
  BlockArgument writeD = taskBody.getArgument(2);
  Value outerM = hyperblockBody.getArgument(0);
  Value outerN = hyperblockBody.getArgument(1);
  memref::LoadOp loadT;
  memref::LoadOp loadE;
  memref::StoreOp storeD;
  arith::AddIOp add;
  unsigned tLoads = 0;
  unsigned eLoads = 0;
  unsigned dStores = 0;
  unsigned addCount = 0;
  consumer.walk([&](memref::LoadOp load) {
    if (load.getMemRef() == readT &&
        equalValues(load.getIndices(), {outerM, outerN})) {
      loadT = load;
      ++tLoads;
      return;
    }
    if (load.getMemRef() == readE &&
        equalValues(load.getIndices(), {outerM, outerN})) {
      loadE = load;
      ++eLoads;
      return;
    }
    error = "UNKNOWN_ACCESS: consumer has a load outside exact T/E "
            "elementwise access";
  });
  consumer.walk([&](memref::StoreOp store) {
    if (store.getMemRef() == writeD &&
        equalValues(store.getIndices(), {outerM, outerN})) {
      storeD = store;
      ++dStores;
      return;
    }
    error = "UNKNOWN_ACCESS: consumer has a store outside exact D access";
  });
  consumer.walk([&](arith::AddIOp op) {
    add = op;
    ++addCount;
  });
  if (!error.empty())
    return false;
  if (tLoads != 1 || eLoads != 1 || dStores != 1 || addCount != 1 || !loadT ||
      !loadE || !storeD || !add || add->getBlock() != &hyperblockBody ||
      loadT->getBlock() != &hyperblockBody ||
      loadE->getBlock() != &hyperblockBody ||
      storeD->getBlock() != &hyperblockBody ||
      !containsValue(add.getOperands(), loadT.getResult()) ||
      !containsValue(add.getOperands(), loadE.getResult()) ||
      storeD.getValue() != add.getResult()) {
    error = "UNKNOWN_ACCESS: consumer does not match the canonical T+E body";
    return false;
  }
  return true;
}

struct SourceContract {
  func::FuncOp function;
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  std::string moduleSha256;
};

static FailureOr<SourceContract>
recognizeSource(ModuleOp module, func::FuncOp function, std::string &error) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  if (tasks.size() != 2) {
    error = "TASK_GRAPH_SHAPE: canonical source requires exactly one "
            "producer and one consumer task";
    return failure();
  }
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  for (TaskflowTaskOp task : tasks) {
    if (task.getTaskName() == "producer")
      producer = task;
    else if (task.getTaskName() == "consumer")
      consumer = task;
  }
  if (!producer || !consumer || producer == consumer) {
    error = "TASK_GRAPH_SHAPE: tasks must be named producer and consumer";
    return failure();
  }
  if (!producer->isBeforeInBlock(consumer)) {
    error = "TASK_GRAPH_SHAPE: producer must precede consumer";
    return failure();
  }

  if (function.getNumArguments() != 5 ||
      function.getFunctionType().getNumResults() != 1 ||
      !isEightByEightI32Type(function.getResultTypes().front())) {
    error = "STATIC_DOMAIN_INVALID: canonical function requires five 8x8 i32 "
            "arguments and one 8x8 i32 result";
    return failure();
  }
  for (unsigned index = 0; index < 5; ++index) {
    if (!isEightByEightI32(function.getArgument(index)) ||
        !hasNoAliasAttr(function, index)) {
      error = "ALIAS_PROOF_MISSING: every canonical function buffer must be a "
              "static 8x8 i32 amoeba.noalias argument";
      return failure();
    }
  }
  Value A = function.getArgument(0);
  Value B = function.getArgument(1);
  Value C = function.getArgument(2);
  Value E = function.getArgument(3);
  Value D = function.getArgument(4);

  if (producer.getWillReads().size() != 3 ||
      producer.getWillWrites().size() != 1 ||
      !producer.getValueInputs().empty() ||
      producer.getOriginalReadMemrefs().size() != 3 ||
      producer.getOriginalWriteMemrefs().size() != 1 ||
      consumer.getWillReads().size() != 2 ||
      consumer.getWillWrites().size() != 1 ||
      !consumer.getValueInputs().empty() ||
      consumer.getOriginalReadMemrefs().size() != 2 ||
      consumer.getOriginalWriteMemrefs().size() != 1 ||
      producer.getDoneWrites().size() != 1 ||
      consumer.getDoneWrites().size() != 1) {
    error = "TASK_GRAPH_SHAPE: Taskflow operand segments do not match the "
            "canonical producer/consumer contract";
    return failure();
  }
  if (!equalValues(producer.getWillReads(), {A, B, C}) ||
      !equalValues(producer.getOriginalReadMemrefs(), {A, B, C}) ||
      !equalValues(consumer.getWillReads(),
                   {producer.getDoneWrites().front(), E}) ||
      !equalValues(consumer.getOriginalReadMemrefs(),
                   {producer.getWillWrites().front(), E}) ||
      producer.getOriginalWriteMemrefs().front() !=
          producer.getWillWrites().front() ||
      consumer.getWillWrites().front() != D ||
      consumer.getOriginalWriteMemrefs().front() != D ||
      producer.getDoneWrites().front().getType() !=
          producer.getWillWrites().front().getType() ||
      consumer.getDoneWrites().front().getType() !=
          consumer.getWillWrites().front().getType()) {
    error = "EDGE_ENDPOINT_INVALID: canonical A/B/C/T/E/D operand identity or "
            "result segments are malformed";
    return failure();
  }
  if (consumer.getWillReads().front() != producer.getDoneWrites().front()) {
    error = "EDGE_ENDPOINT_INVALID: consumer must have one direct raw T "
            "dependency from producer done_writes";
    return failure();
  }

  Value T = producer.getWillWrites().front();
  auto allocation = T.getDefiningOp<memref::AllocOp>();
  if (!allocation ||
      allocation->getParentOp() != function.getBody().getParentOp() ||
      !isEightByEightI32(T)) {
    error = "ALIAS_PROOF_MISSING: T must be one internal 8x8 i32 allocation";
    return failure();
  }
  unsigned producerUses = 0;
  unsigned consumerUses = 0;
  for (OpOperand &use : T.getUses()) {
    if (use.getOwner() == producer.getOperation())
      ++producerUses;
    else if (use.getOwner() == consumer.getOperation())
      ++consumerUses;
    else {
      error = "EXTERNAL_USER_PRESENT: internal T has a use outside the "
              "canonical producer/consumer pair";
      return failure();
    }
  }
  if (producerUses != 2 || consumerUses != 1) {
    error = "EXTERNAL_USER_PRESENT: internal T must have exactly one producer "
            "write and one consumer read";
    return failure();
  }

  if (producer.getDoneWrites().front().getUses().empty() ||
      !llvm::all_of(
          producer.getDoneWrites().front().getUsers(),
          [&](Operation *user) { return user == consumer.getOperation(); })) {
    error = "EDGE_ENDPOINT_INVALID: T done_writes must have exactly one "
            "consumer user";
    return failure();
  }
  if (!checkKnownBodyOperations(producer, error) ||
      !checkKnownBodyOperations(consumer, error))
    return failure();
  TaskflowHyperblockOp producerHyperblock =
      findSingleHyperblock(producer, error);
  if (!producerHyperblock)
    return failure();
  TaskflowHyperblockOp consumerHyperblock =
      findSingleHyperblock(consumer, error);
  if (!consumerHyperblock)
    return failure();
  if (!checkStaticCounterChain(producer, producerHyperblock, 2, error) ||
      !checkStaticCounterChain(consumer, consumerHyperblock, 2, error))
    return failure();
  scf::ForOp kLoop = findSingleKLoop(producer, error);
  if (!kLoop || !checkStaticKLoop(kLoop, error))
    return failure();
  bool consumerHasLoop = false;
  consumer.walk([&](scf::ForOp) { consumerHasLoop = true; });
  if (consumerHasLoop) {
    error = "DYNAMIC_CONTROL: consumer must have no reduction loop";
    return failure();
  }
  if (!checkTaskYield(producer, producer.getWillReads().size(), error) ||
      !checkTaskYield(consumer, consumer.getWillReads().size(), error) ||
      !checkProducerBody(producer, producerHyperblock, kLoop, error) ||
      !checkConsumerBody(consumer, consumerHyperblock, error))
    return failure();

  for (Operation &operation : function.getBody().front()) {
    if (isa<memref::AllocOp, TaskflowTaskOp, func::ReturnOp>(&operation))
      continue;
    error = "SIDE_EFFECT_UNPROVEN: function contains unsupported operation " +
            operation.getName().getStringRef().str();
    return failure();
  }
  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != consumer.getDoneWrites().front()) {
    error = "EDGE_ENDPOINT_INVALID: canonical function must return consumer D "
            "done_writes";
    return failure();
  }

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module.print(stream);
  stream.flush();
  return SourceContract{function, producer, consumer, hashBytes(printed)};
}

struct RegionSpec {
  std::string buffer;
  Range lower{0, 0};
  Range upper{0, 0};
  std::string access;
  std::string visibility;
  std::string ownership;
  std::string logicalTile;
  std::string logicalKBlock;
  bool completionOnly = false;
};

static RegionSpec region(StringRef buffer, Range lower, Range upper,
                         StringRef access) {
  RegionSpec result;
  result.buffer = buffer.str();
  result.lower = lower;
  result.upper = upper;
  result.access = access.str();
  return result;
}

static llvm::json::Object regionJson(const RegionSpec &region) {
  llvm::json::Object result{
      {"schema", kRegionSchema},
      {"buffer", region.buffer},
      {"lower", llvm::json::Array{region.lower[0], region.lower[1]}},
      {"upper", llvm::json::Array{region.upper[0], region.upper[1]}},
      {"half_open", true},
      {"access", region.access}};
  if (!region.visibility.empty())
    result["visibility"] = region.visibility;
  if (!region.ownership.empty())
    result["ownership"] = region.ownership;
  if (!region.logicalTile.empty())
    result["logical_output_tile_id"] = region.logicalTile;
  if (!region.logicalKBlock.empty())
    result["logical_k_block"] = region.logicalKBlock;
  if (region.completionOnly)
    result["completion_only"] = true;
  return result;
}

struct TaskSpec {
  std::string id;
  std::string kind;
  std::string role;
  std::string tile;
  Range m{0, 0};
  Range n{0, 0};
  std::optional<Range> k;
  std::map<std::string, std::vector<RegionSpec>> regions;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::vector<std::string> predecessors;
  std::vector<std::string> successors;
  std::vector<std::string> body;
  std::string logicalKBlock;
  std::string outputVisibility;
  std::string initialCOwnership;
  std::string finalCOwnership;
  std::string partialBuffer;
  bool numericReduction = false;
  bool completionOnly = false;
};

struct EdgeSpec {
  std::string source;
  std::string target;
  std::string kind;
  std::string scope;
  std::optional<RegionSpec> transfer;
};

enum class KPolicy { Sequential, Parallel };
enum class FusionMode {
  None,
  Whole,
  TileLocal,
  SequentialLast,
  ReductionConsumer
};

struct GraphState {
  int64_t mFactor = 0;
  int64_t nFactor = 0;
  int64_t kFactor = 0;
  KPolicy kPolicy = KPolicy::Sequential;
  std::string reductionTopology = "none";
  FusionMode fusion = FusionMode::None;
  bool numericReduction = false;
  bool completionJoin = false;
  bool reductionConsumerIntent = false;
};

static StringRef kPolicyName(KPolicy policy) {
  return policy == KPolicy::Parallel ? "parallel" : "sequential";
}

static StringRef fusionName(FusionMode mode) {
  switch (mode) {
  case FusionMode::None:
    return "none";
  case FusionMode::Whole:
    return "whole";
  case FusionMode::TileLocal:
    return "tile_local";
  case FusionMode::SequentialLast:
    return "sequential_last";
  case FusionMode::ReductionConsumer:
    return "reduction_consumer";
  }
  return "none";
}

static std::string tileId(Range m, Range n) {
  return "tile_m" + std::to_string(m[0]) + "_n" + std::to_string(n[0]);
}

static std::string kBlockId(unsigned index, Range k) {
  return "k" + std::to_string(index) + "_" + std::to_string(k[0]) + "_" +
         std::to_string(k[1]);
}

static SmallVector<Range> splitRange(int64_t factor) {
  SmallVector<Range> result;
  if (factor == 0) {
    result.push_back({0, kDimension});
    return result;
  }
  for (int64_t start = 0; start < kDimension; start += factor)
    result.push_back({start, std::min<int64_t>(start + factor, kDimension)});
  return result;
}

static void appendUnique(std::vector<std::string> &values, StringRef value) {
  if (!llvm::is_contained(values, value.str()))
    values.push_back(value.str());
}

static void deriveTaskBuffers(TaskSpec &task) {
  for (const auto &entry : task.regions)
    for (const RegionSpec &item : entry.second) {
      if (item.visibility == "fused_internal" &&
          item.buffer == "fused_internal_T")
        continue;
      if (item.access == "read" || item.access == "private_read" ||
          item.access == "read_write")
        appendUnique(task.reads, item.buffer);
      if (item.access == "write" || item.access == "private_write" ||
          item.access == "read_write")
        appendUnique(task.writes, item.buffer);
    }
}

static void addRegion(TaskSpec &task, StringRef name, RegionSpec item) {
  task.regions[name.str()].push_back(std::move(item));
}

static std::string stateSignature(const SourceContract &source,
                                  const GraphState &state) {
  std::string result = source.moduleSha256;
  result += "|M=" + std::to_string(state.mFactor);
  result += "|N=" + std::to_string(state.nFactor);
  result += "|K=" + std::to_string(state.kFactor);
  result += "|k_policy=" + kPolicyName(state.kPolicy).str();
  result += "|topology=" + state.reductionTopology;
  result += "|fusion=" + fusionName(state.fusion).str();
  result += "|numeric=" + std::to_string(state.numericReduction);
  result += "|completion=" + std::to_string(state.completionJoin);
  result += "|intent=" + std::to_string(state.reductionConsumerIntent);
  return result;
}

static void addTaskEdges(TaskSpec &task, const std::vector<EdgeSpec> &edges) {
  for (const EdgeSpec &edge : edges) {
    if (edge.target == task.id)
      task.predecessors.push_back(edge.source);
    if (edge.source == task.id)
      task.successors.push_back(edge.target);
  }
  llvm::sort(task.predecessors);
  llvm::sort(task.successors);
}

struct BuiltGraph {
  GraphState state;
  std::string graphId;
  std::vector<TaskSpec> tasks;
  std::vector<EdgeSpec> edges;
  std::vector<std::string> terminals;
};

static BuiltGraph buildGraph(const SourceContract &source,
                             const GraphState &state, int64_t tileFactor) {
  (void)tileFactor;
  BuiltGraph graph;
  graph.state = state;
  SmallVector<Range> mTiles = splitRange(state.mFactor);
  SmallVector<Range> nTiles = splitRange(state.nFactor);
  SmallVector<Range> kBlocks = splitRange(state.kFactor);
  bool parallel = state.kPolicy == KPolicy::Parallel;
  for (Range mRange : mTiles) {
    for (Range nRange : nTiles) {
      std::string outputTile = tileId(mRange, nRange);
      std::vector<std::string> tileProducers;
      if (parallel) {
        for (auto indexed : llvm::enumerate(kBlocks)) {
          Range kRange = indexed.value();
          std::string kid = kBlockId(indexed.index(), kRange);
          std::string partial = "partial_T/" + outputTile + "/" + kid;
          TaskSpec task{"partial_" + outputTile + "_" + kid,
                        "gemm_producer",
                        "producer",
                        outputTile,
                        mRange,
                        nRange,
                        kRange};
          addRegion(task, "A_read",
                    region("A", {mRange[0], kRange[0]}, {mRange[1], kRange[1]},
                           "read"));
          addRegion(task, "B_read",
                    region("B", {kRange[0], nRange[0]}, {kRange[1], nRange[1]},
                           "read"));
          RegionSpec partialRegion =
              region(partial, {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                     "private_write");
          partialRegion.logicalTile = outputTile;
          partialRegion.logicalKBlock = kid;
          addRegion(task, "partial_output", std::move(partialRegion));
          task.logicalKBlock = kid;
          task.outputVisibility = "private_partial";
          task.initialCOwnership = "reduction_task";
          task.finalCOwnership = "private_partial";
          task.partialBuffer = partial;
          task.body = {"load_A", "load_B", "mul", "store_partial"};
          deriveTaskBuffers(task);
          graph.tasks.push_back(std::move(task));
          tileProducers.push_back(graph.tasks.back().id);
        }

        // A balanced numeric reduction materializes one explicit reduction
        // task per internal tree node before its final reduction.  Keep those
        // nodes in the semantic graph: a tree is not equivalent to the
        // linear reduction when the graph facts consumer compares task and
        // edge counts.  The standalone K pass names these nodes
        // ``reduction.tree.<level>.<pair>``; the graph IDs remain stable
        // synthetic IDs, just like the producer and consumer tile IDs.
        struct ReductionInput {
          std::string taskId;
          std::string buffer;
          Range k;
        };
        std::vector<ReductionInput> reductionInputs;
        reductionInputs.reserve(tileProducers.size());
        for (auto indexed : llvm::enumerate(kBlocks)) {
          Range kRange = indexed.value();
          std::string kid = kBlockId(indexed.index(), kRange);
          reductionInputs.push_back({tileProducers[indexed.index()],
                                     "partial_T/" + outputTile + "/" + kid,
                                     kRange});
        }
        if (state.reductionTopology == "balanced_tree") {
          unsigned level = 0;
          while (reductionInputs.size() > 1) {
            std::vector<ReductionInput> nextInputs;
            nextInputs.reserve((reductionInputs.size() + 1) / 2);
            for (unsigned index = 0; index < reductionInputs.size();
                 index += 2) {
              if (index + 1 == reductionInputs.size()) {
                nextInputs.push_back(std::move(reductionInputs[index]));
                continue;
              }
              const ReductionInput &lhs = reductionInputs[index];
              const ReductionInput &rhs = reductionInputs[index + 1];
              std::string treeId = "reduction_tree_" + outputTile + "_l" +
                                   std::to_string(level) + "_p" +
                                   std::to_string(index / 2);
              std::string treeBuffer = "partial_T/" + outputTile + "/tree." +
                                       std::to_string(level) + "." +
                                       std::to_string(index / 2);
              Range treeK{lhs.k[0], rhs.k[1]};
              TaskSpec tree{treeId, "reduction", "reduction", outputTile,
                            mRange, nRange,      treeK};
              RegionSpec lhsInput =
                  region(lhs.buffer, {mRange[0], nRange[0]},
                         {mRange[1], nRange[1]}, "private_read");
              lhsInput.logicalTile = outputTile;
              lhsInput.logicalKBlock = kBlockId(0, lhs.k);
              tree.regions["partial_inputs"].push_back(std::move(lhsInput));
              RegionSpec rhsInput =
                  region(rhs.buffer, {mRange[0], nRange[0]},
                         {mRange[1], nRange[1]}, "private_read");
              rhsInput.logicalTile = outputTile;
              rhsInput.logicalKBlock = kBlockId(0, rhs.k);
              tree.regions["partial_inputs"].push_back(std::move(rhsInput));
              RegionSpec treeOutput =
                  region(treeBuffer, {mRange[0], nRange[0]},
                         {mRange[1], nRange[1]}, "private_write");
              treeOutput.logicalTile = outputTile;
              treeOutput.logicalKBlock = "tree." + std::to_string(level) + "." +
                                         std::to_string(index / 2);
              addRegion(tree, "partial_output", std::move(treeOutput));
              tree.outputVisibility = "global_materialized";
              tree.initialCOwnership = "reduction_task";
              tree.finalCOwnership = "private_partial";
              tree.partialBuffer = treeBuffer;
              tree.numericReduction = true;
              tree.body = {"reduce_private_partials", "store_partial"};
              deriveTaskBuffers(tree);
              graph.tasks.push_back(std::move(tree));
              graph.edges.push_back(
                  {lhs.taskId, treeId, "partial_reduction", "tile_local",
                   region(lhs.buffer, {mRange[0], nRange[0]},
                          {mRange[1], nRange[1]}, "private_read")});
              graph.edges.push_back(
                  {rhs.taskId, treeId, "partial_reduction", "tile_local",
                   region(rhs.buffer, {mRange[0], nRange[0]},
                          {mRange[1], nRange[1]}, "private_read")});
              nextInputs.push_back({treeId, treeBuffer, treeK});
            }
            reductionInputs = std::move(nextInputs);
            ++level;
          }
        }
        std::string reduceId = "reduce_" + outputTile;
        bool fused = state.fusion == FusionMode::ReductionConsumer;
        TaskSpec reduce{reduceId,
                        fused ? "fused_reduction_consumer" : "reduction",
                        fused ? "fused_reduction_consumer" : "reduction",
                        outputTile,
                        mRange,
                        nRange,
                        Range{0, kDimension}};
        for (const ReductionInput &reductionInput : reductionInputs) {
          RegionSpec inputRegion =
              region(reductionInput.buffer, {mRange[0], nRange[0]},
                     {mRange[1], nRange[1]}, "private_read");
          inputRegion.logicalTile = outputTile;
          inputRegion.logicalKBlock = kBlockId(0, reductionInput.k);
          reduce.regions["partial_inputs"].push_back(std::move(inputRegion));
          graph.edges.push_back(
              {reductionInput.taskId, reduceId, "partial_reduction",
               "tile_local",
               region(reductionInput.buffer, {mRange[0], nRange[0]},
                      {mRange[1], nRange[1]}, "private_read")});
        }
        addRegion(reduce, "C_initial_input",
                  region("C_initial", {mRange[0], nRange[0]},
                         {mRange[1], nRange[1]}, "read"));
        RegionSpec reduceOutput =
            region(fused ? "D" : "T", {mRange[0], nRange[0]},
                   {mRange[1], nRange[1]}, "write");
        reduceOutput.visibility = fused ? "fused_internal" : "global";
        addRegion(reduce, "T_output", std::move(reduceOutput));
        reduce.outputVisibility =
            fused ? "fused_internal" : "global_materialized";
        reduce.initialCOwnership = "initial_C_once";
        reduce.finalCOwnership = fused ? "external_final_D" : "global_T";
        reduce.numericReduction = true;
        reduce.body = fused
                          ? std::vector<std::string>{"reduce_private_partials",
                                                     "add_E", "store_D"}
                          : std::vector<std::string>{"reduce_private_partials",
                                                     "store_T"};
        deriveTaskBuffers(reduce);
        graph.tasks.push_back(std::move(reduce));
        if (!fused) {
          std::string consumerId = "consumer_" + outputTile;
          TaskSpec consumer{consumerId,  "elementwise_consumer",
                            "consumer",  outputTile,
                            mRange,      nRange,
                            std::nullopt};
          addRegion(consumer, "T_read",
                    region("T", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                           "read"));
          addRegion(consumer, "E_read",
                    region("E", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                           "read"));
          addRegion(consumer, "D_output",
                    region("D", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                           "write"));
          consumer.outputVisibility = "global_materialized";
          consumer.finalCOwnership = "external_final_D";
          consumer.body = {"load_T", "load_E", "add", "store_D"};
          deriveTaskBuffers(consumer);
          graph.tasks.push_back(std::move(consumer));
          graph.edges.push_back({reduceId, consumerId, "producer_consumer",
                                 "tile_local",
                                 region("T", {mRange[0], nRange[0]},
                                        {mRange[1], nRange[1]}, "read")});
          graph.terminals.push_back(consumerId);
        } else {
          graph.terminals.push_back(reduceId);
        }
        continue;
      }

      std::string previous;
      for (auto indexed : llvm::enumerate(kBlocks)) {
        Range kRange = indexed.value();
        std::string kid = kBlockId(indexed.index(), kRange);
        bool last = indexed.index() + 1 == kBlocks.size();
        bool fused = state.fusion == FusionMode::Whole ||
                     (state.fusion == FusionMode::TileLocal) ||
                     (state.fusion == FusionMode::SequentialLast && last);
        if (state.fusion == FusionMode::Whole && kBlocks.size() > 1)
          fused = last;
        std::string taskId =
            (fused ? "fused_" : "producer_") + outputTile + "_" + kid;
        TaskSpec task{taskId,
                      fused ? "fused_producer_consumer" : "gemm_producer",
                      fused ? "fused_producer_consumer" : "producer",
                      outputTile,
                      mRange,
                      nRange,
                      kRange};
        addRegion(task, "A_read",
                  region("A", {mRange[0], kRange[0]}, {mRange[1], kRange[1]},
                         "read"));
        addRegion(task, "B_read",
                  region("B", {kRange[0], nRange[0]}, {kRange[1], nRange[1]},
                         "read"));
        RegionSpec cInput =
            region(indexed.index() == 0 ? "C_initial" : "state_carry",
                   {mRange[0], nRange[0]}, {mRange[1], nRange[1]}, "read");
        cInput.ownership =
            indexed.index() == 0 ? "initial_C_once" : "previous_K_block";
        addRegion(task, "C_input", std::move(cInput));
        RegionSpec output = region(fused ? "D" : "T", {mRange[0], nRange[0]},
                                   {mRange[1], nRange[1]}, "write");
        output.visibility = fused ? "fused_internal" : "global";
        addRegion(task, "T_output", std::move(output));
        if (fused) {
          RegionSpec tRead = region("fused_internal_T", {mRange[0], nRange[0]},
                                    {mRange[1], nRange[1]}, "read");
          tRead.visibility = "fused_internal";
          addRegion(task, "T_read", std::move(tRead));
          addRegion(task, "E_read",
                    region("E", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                           "read"));
          addRegion(task, "D_output",
                    region("D", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                           "write"));
        }
        task.logicalKBlock = kid;
        task.outputVisibility =
            fused ? "fused_internal" : "global_materialized";
        task.initialCOwnership =
            indexed.index() == 0 ? "initial_C_once" : "state_carry";
        task.finalCOwnership =
            fused ? "external_final_D"
                  : (last ? "external_final_C" : "state_carry");
        task.body =
            fused ? std::vector<std::string>{"load_A",      "load_B",
                                             "mul",         "load_C_or_state",
                                             "add_product", "load_E",
                                             "add_E",       "store_D"}
                  : std::vector<std::string>{"load_A",      "load_B",
                                             "mul",         "load_C_or_state",
                                             "add_product", "store_T"};
        deriveTaskBuffers(task);
        graph.tasks.push_back(std::move(task));
        tileProducers.push_back(graph.tasks.back().id);
        if (!previous.empty())
          graph.edges.push_back({previous, taskId, "state_carry", "tile_local",
                                 region("state_carry", {mRange[0], nRange[0]},
                                        {mRange[1], nRange[1]}, "read_write")});
        previous = taskId;
      }
      bool finalFused = !graph.tasks.empty() &&
                        graph.tasks.back().id == previous &&
                        graph.tasks.back().kind == "fused_producer_consumer";
      if (!finalFused) {
        std::string consumerId = "consumer_" + outputTile;
        TaskSpec consumer{consumerId,  "elementwise_consumer",
                          "consumer",  outputTile,
                          mRange,      nRange,
                          std::nullopt};
        addRegion(consumer, "T_read",
                  region("T", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                         "read"));
        addRegion(consumer, "E_read",
                  region("E", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                         "read"));
        addRegion(consumer, "D_output",
                  region("D", {mRange[0], nRange[0]}, {mRange[1], nRange[1]},
                         "write"));
        consumer.outputVisibility = "global_materialized";
        consumer.finalCOwnership = "external_final_D";
        consumer.body = {"load_T", "load_E", "add", "store_D"};
        deriveTaskBuffers(consumer);
        graph.tasks.push_back(std::move(consumer));
        graph.edges.push_back({previous, consumerId, "producer_consumer",
                               "tile_local",
                               region("T", {mRange[0], nRange[0]},
                                      {mRange[1], nRange[1]}, "read")});
        graph.terminals.push_back(consumerId);
      } else {
        graph.terminals.push_back(previous);
      }
    }
  }

  if (state.completionJoin) {
    TaskSpec join{"completion_join", "completion_join", "completion_join",
                  "all_outputs",     {0, kDimension},   {0, kDimension},
                  std::nullopt};
    RegionSpec completion =
        region("D", {0, 0}, {kDimension, kDimension}, "completion");
    completion.completionOnly = true;
    join.regions["D_output"].push_back(std::move(completion));
    join.outputVisibility = "external_final_D";
    join.completionOnly = true;
    join.body = {"completion_only"};
    for (const std::string &terminal : graph.terminals)
      graph.edges.push_back(
          {terminal, join.id, "completion", "tensor_wide",
           region("D", {0, 0}, {kDimension, kDimension}, "completion")});
    graph.tasks.push_back(std::move(join));
  }
  for (TaskSpec &task : graph.tasks)
    addTaskEdges(task, graph.edges);

  std::string signature = stateSignature(source, state);
  for (const TaskSpec &task : graph.tasks) {
    signature += "|task=" + task.id + ":" + task.kind + ":" + task.role;
    signature +=
        ":m" + std::to_string(task.m[0]) + "," + std::to_string(task.m[1]);
    signature +=
        ":n" + std::to_string(task.n[0]) + "," + std::to_string(task.n[1]);
    if (task.k)
      signature += ":k" + std::to_string((*task.k)[0]) + "," +
                   std::to_string((*task.k)[1]);
    for (const auto &entry : task.regions) {
      signature += ":r=" + entry.first;
      for (const RegionSpec &item : entry.second)
        signature += ":" + item.buffer + ":" + item.access + ":" +
                     std::to_string(item.lower[0]) + "," +
                     std::to_string(item.lower[1]) + ":" +
                     std::to_string(item.upper[0]) + "," +
                     std::to_string(item.upper[1]);
    }
  }
  for (const EdgeSpec &edge : graph.edges)
    signature += "|edge=" + edge.source + ">" + edge.target + ":" + edge.kind +
                 ":" + edge.scope;
  graph.graphId = "graph-" + hashBytes(signature);
  return graph;
}

static llvm::json::Array rangeJson(Range range) {
  return llvm::json::Array{range[0], range[1]};
}

static llvm::json::Value regionsJson(const std::vector<RegionSpec> &regions) {
  if (regions.size() == 1)
    return llvm::json::Value(regionJson(regions.front()));
  llvm::json::Array result;
  for (const RegionSpec &region : regions)
    result.push_back(regionJson(region));
  return llvm::json::Value(std::move(result));
}

static llvm::json::Array stringArray(ArrayRef<std::string> values);

static llvm::json::Object taskJson(const TaskSpec &task,
                                   const GraphState &state) {
  llvm::json::Object regions;
  for (const auto &entry : task.regions)
    regions[entry.first] = regionsJson(entry.second);
  llvm::json::Array reads;
  for (StringRef value : task.reads)
    reads.push_back(value.str());
  llvm::json::Array writes;
  for (StringRef value : task.writes)
    writes.push_back(value.str());
  llvm::json::Array predecessors;
  for (StringRef value : task.predecessors)
    predecessors.push_back(value.str());
  llvm::json::Array successors;
  for (StringRef value : task.successors)
    successors.push_back(value.str());
  llvm::json::Array body;
  for (StringRef value : task.body)
    body.push_back(value.str());
  llvm::json::Object result{
      {"task_id", task.id},
      {"task_kind", task.kind},
      {"task_role", task.role},
      {"semantic_task", true},
      {"logical_output_tile_id", task.tile},
      {"m_range", rangeJson(task.m)},
      {"n_range", rangeJson(task.n)},
      {"M_range", rangeJson(task.m)},
      {"N_range", rangeJson(task.n)},
      {"k_range", task.k ? llvm::json::Value(rangeJson(*task.k))
                         : llvm::json::Value(nullptr)},
      {"K_range", task.k ? llvm::json::Value(rangeJson(*task.k))
                         : llvm::json::Value(nullptr)},
      {"regions", std::move(regions)},
      {"reads", std::move(reads)},
      {"writes", std::move(writes)},
      {"will_reads", stringArray(task.reads)},
      {"will_writes", stringArray(task.writes)},
      {"original_read_memrefs", stringArray(task.reads)},
      {"original_write_memrefs", stringArray(task.writes)},
      {"k_policy", kPolicyName(state.kPolicy)},
      {"reduction_topology", state.reductionTopology},
      {"intermediate_visibility", task.outputVisibility},
      {"output_visibility", task.outputVisibility},
      {"predecessors", std::move(predecessors)},
      {"successors", std::move(successors)},
      {"body", std::move(body)},
      {"numeric_reduction", task.numericReduction},
      {"completion_only", task.completionOnly}};
  if (!task.logicalKBlock.empty())
    result["logical_k_block_id"] = task.logicalKBlock;
  if (!task.initialCOwnership.empty())
    result["initial_C_ownership"] = task.initialCOwnership;
  if (!task.finalCOwnership.empty())
    result["final_C_ownership"] = task.finalCOwnership;
  if (!task.partialBuffer.empty())
    result["partial_buffer"] = task.partialBuffer;
  if (task.role == "producer")
    result["source_task_name"] = "producer";
  else if (task.role == "consumer")
    result["source_task_name"] = "consumer";
  else if (task.role == "fused_producer_consumer" ||
           task.role == "fused_reduction_consumer")
    result["source_task_names"] = llvm::json::Array{"producer", "consumer"};
  return result;
}

static llvm::json::Object edgeJson(const EdgeSpec &edge,
                                   ArrayRef<TaskSpec> tasks) {
  llvm::json::Object result{
      {"source_task", edge.source}, {"target_task", edge.target},
      {"kind", edge.kind},          {"type", edge.kind},
      {"scope", edge.scope},        {"edge_scope", edge.scope}};
  for (const TaskSpec &task : tasks) {
    if (task.id == edge.source) {
      result["source_task_role"] = task.role;
      if (task.role == "producer")
        result["source_task_name"] = "producer";
      else if (task.role == "consumer")
        result["source_task_name"] = "consumer";
    }
    if (task.id == edge.target) {
      result["target_task_role"] = task.role;
      if (task.role == "producer")
        result["target_task_name"] = "producer";
      else if (task.role == "consumer")
        result["target_task_name"] = "consumer";
    }
  }
  if (edge.transfer)
    result["region"] = regionJson(*edge.transfer);
  return result;
}

static llvm::json::Object graphJson(const SourceContract &source,
                                    const BuiltGraph &graph,
                                    int64_t tileFactor) {
  llvm::json::Object identity{
      {"schema", "orbit-canonical-producer-consumer-pipeline-v1"},
      {"canonical_pipeline_id", "canonical-producer-consumer-8x8x8-i32-v1"},
      {"canonical_source_id", source.moduleSha256},
      {"source_module_sha256", source.moduleSha256},
      {"source_module_sha256_kind", "normalized_mlir_print"},
      {"source_module_sha256_exact_input_bytes", false},
      {"function",
       source.function->getAttrOfType<StringAttr>("sym_name").getValue().str()},
      {"M", kDimension},
      {"N", kDimension},
      {"K", kDimension},
      {"dtype", "i32"},
      {"layout", "row_major_identity"},
      {"arithmetic_semantics", kArithmeticSemantics},
      {"alias_proof",
       "five distinct amoeba.noalias arguments plus internal T allocation"}};
  llvm::json::Array tasks;
  for (const TaskSpec &task : graph.tasks)
    tasks.push_back(taskJson(task, graph.state));
  llvm::json::Array edges;
  for (const EdgeSpec &edge : graph.edges)
    edges.push_back(edgeJson(edge, graph.tasks));
  llvm::json::Array mTiles;
  for (Range range : splitRange(graph.state.mFactor))
    mTiles.push_back(rangeJson(range));
  llvm::json::Array nTiles;
  for (Range range : splitRange(graph.state.nFactor))
    nTiles.push_back(rangeJson(range));
  llvm::json::Array kBlocks;
  for (Range range : splitRange(graph.state.kFactor))
    kBlocks.push_back(rangeJson(range));
  llvm::json::Array terminals;
  for (StringRef terminal : graph.terminals)
    terminals.push_back(terminal.str());
  llvm::json::Array groups;
  if (graph.state.fusion == FusionMode::Whole)
    groups.push_back("producer_consumer");
  else if (graph.state.fusion == FusionMode::TileLocal) {
    for (const TaskSpec &task : graph.tasks)
      if (task.role == "fused_producer_consumer")
        groups.push_back("producer_consumer:" + task.tile);
  } else if (graph.state.fusion == FusionMode::SequentialLast) {
    groups.push_back("producer_consumer:last_k");
  } else if (graph.state.fusion == FusionMode::ReductionConsumer) {
    groups.push_back("reduction_consumer");
  }
  llvm::json::Object fusion{
      {"producer_consumer", graph.state.fusion == FusionMode::Whole
                                ? llvm::json::Value("whole")
                            : graph.state.fusion == FusionMode::TileLocal
                                ? llvm::json::Value("tile_local")
                                : llvm::json::Value("none")},
      {"sequential_last", graph.state.fusion == FusionMode::SequentialLast},
      {"reduction_consumer",
       graph.state.fusion == FusionMode::ReductionConsumer},
      {"groups", std::move(groups)}};
  std::string intermediateVisibility = graph.state.fusion == FusionMode::None
                                           ? "global_materialized"
                                           : "fused_internal";
  llvm::json::Object intermediate{
      {"identity", "T"},
      {"visibility", intermediateVisibility},
      {"state", intermediateVisibility == "fused_internal"
                    ? llvm::json::Value("eliminated")
                    : llvm::json::Value("global_materialized")},
      {"allocation", intermediateVisibility == "fused_internal"
                         ? llvm::json::Value("eliminated")
                         : llvm::json::Value("global_T")},
      {"lifetime", intermediateVisibility == "fused_internal"
                       ? llvm::json::Value("within_fused_task")
                       : llvm::json::Value("producer_to_consumer")},
      {"external_users", llvm::json::Array{}},
      {"private_partial", graph.state.kPolicy == KPolicy::Parallel},
      {"producer", "producer"},
      {"consumer", "consumer"}};
  llvm::json::Array partialBuffers;
  for (const TaskSpec &task : graph.tasks)
    if (!task.partialBuffer.empty())
      partialBuffers.push_back(task.partialBuffer);
  intermediate["partial_buffers"] = std::move(partialBuffers);
  std::string primary = "identity";
  if (graph.state.fusion != FusionMode::None)
    primary = "fusion_only";
  if (graph.state.mFactor || graph.state.nFactor || graph.state.kFactor) {
    primary = graph.state.fusion == FusionMode::None ? "tiling_only"
                                                     : "tiling_and_fusion";
  }
  llvm::json::Array modeTags;
  modeTags.push_back(primary);
  if (graph.state.kPolicy == KPolicy::Parallel && graph.state.numericReduction)
    modeTags.push_back(graph.state.reductionTopology + "_reduction");
  if (graph.state.fusion == FusionMode::SequentialLast)
    modeTags.push_back("sequential_last_fusion");
  if (graph.state.fusion == FusionMode::ReductionConsumer)
    modeTags.push_back("reduction_consumer_fusion");
  return llvm::json::Object{
      {"schema", kGraphSchema},
      {"graph_kind", "semantic_task_graph"},
      {"graph_id", graph.graphId},
      {"variant_id", graph.graphId},
      {"canonical_source_id", source.moduleSha256},
      {"source_module_sha256", source.moduleSha256},
      {"source_module_sha256_kind", "normalized_mlir_print"},
      {"source_module_sha256_exact_input_bytes", false},
      {"source_identity", std::move(identity)},
      {"dimensions", llvm::json::Object{{"M", kDimension},
                                        {"N", kDimension},
                                        {"K", kDimension}}},
      {"M", kDimension},
      {"N", kDimension},
      {"K", kDimension},
      {"dtype", "i32"},
      {"layout", "row_major_identity"},
      {"arithmetic_semantics", kArithmeticSemantics},
      {"tail_policy", "allow"},
      {"tile_factor", tileFactor},
      {"m_tiles", std::move(mTiles)},
      {"n_tiles", std::move(nTiles)},
      {"k_blocks", std::move(kBlocks)},
      {"k_policy", kPolicyName(graph.state.kPolicy)},
      {"reduction_topology", graph.state.reductionTopology},
      {"numeric_reduction_materialized", graph.state.numericReduction},
      {"completion_join_materialized", graph.state.completionJoin},
      {"fusion", std::move(fusion)},
      {"intermediate", std::move(intermediate)},
      {"tasks", std::move(tasks)},
      {"edges", std::move(edges)},
      {"terminal_task_ids", std::move(terminals)},
      {"completion_task_id", graph.state.completionJoin
                                 ? llvm::json::Value("completion_join")
                                 : llvm::json::Value(nullptr)},
      {"tile_factor_filter", tileFactor},
      {"tile_factors",
       llvm::json::Object{
           {"M", graph.state.mFactor ? graph.state.mFactor : 1},
           {"N", graph.state.nFactor ? graph.state.nFactor : 1},
           {"K", graph.state.kFactor ? graph.state.kFactor : 1}}},
      {"tile_sizes",
       llvm::json::Object{
           {"BM", graph.state.mFactor ? graph.state.mFactor : kDimension},
           {"BN", graph.state.nFactor ? graph.state.nFactor : kDimension},
           {"BK", graph.state.kFactor ? graph.state.kFactor : kDimension}}},
      {"semantic_legal", true},
      {"mode_tags", llvm::json::Object{{"primary", primary},
                                       {"tags", std::move(modeTags)}}}};
}

struct ActionSpec {
  std::string kind;
  bool tiled = false;
  std::string pass;
  int64_t tileSize = 0;
  bool executable = true;
  std::string materializationStatus = "implemented";
};

static llvm::json::Object actionJson(const ActionSpec &action);

static llvm::json::Object actionWitnessJson(const ActionSpec &action) {
  return actionJson(action);
}

static llvm::json::Object witnessJson(ArrayRef<ActionSpec> history) {
  llvm::json::Array actions;
  llvm::json::Array pipeline;
  bool executable = true;
  std::string materializationStatus = "implemented";
  for (const ActionSpec &action : history) {
    actions.push_back(actionWitnessJson(action));
    std::string step = action.pass;
    if (action.tiled)
      step += "{tile-size=" + std::to_string(action.tileSize) + "}";
    pipeline.push_back(step);
    executable &= action.executable;
    if (!action.executable && materializationStatus == "implemented")
      materializationStatus = action.materializationStatus;
  }
  return llvm::json::Object{{"actions", std::move(actions)},
                            {"pass_pipeline", std::move(pipeline)},
                            {"executable", executable},
                            {"materialization_status", materializationStatus}};
}

static SmallVector<ActionSpec> actionCatalog(int64_t tileFactor) {
  // This catalog records pass availability.  actionForHistory checks whether
  // each available pass can replay at a particular point in an action history.
  SmallVector<ActionSpec> result;
  result.push_back({"FuseProducerConsumer", false, "fuse-producer-consumer", 0,
                    true, "implemented"});
  SmallVector<int64_t> factors;
  if (tileFactor == 0)
    factors = {2, 4, 8};
  else if (tileFactor == 1)
    factors = {};
  else
    factors = {tileFactor};
  for (int64_t factor : factors) {
    bool executable = factor < kDimension;
    StringRef status = executable ? "implemented" : "tile_size_is_full_domain";
    result.push_back({"TileOutputM", true, "tile-output-m", factor, executable,
                      status.str()});
  }
  for (int64_t factor : factors) {
    bool executable = factor < kDimension;
    StringRef status = executable ? "implemented" : "tile_size_is_full_domain";
    result.push_back({"TileOutputN", true, "tile-output-n", factor, executable,
                      status.str()});
  }
  // A K block equal to the complete reduction domain leaves the source
  // semantics unchanged.  Keep that spelling in the catalog so the manifest
  // records an explicit rejection, but never allow it into a graph state: in
  // particular, the standalone sequential K pass rejects that width and a
  // witness containing it would replay to a different task graph.
  SmallVector<int64_t> kFactors;
  if (tileFactor == 0)
    kFactors = {2, 4, 8};
  else if (tileFactor != 1)
    kFactors = {tileFactor};
  for (int64_t factor : kFactors) {
    bool fullDomain = factor == kDimension;
    result.push_back({"TileReductionKSequential", true,
                      "tile-reduction-k-sequential", factor, !fullDomain,
                      fullDomain ? "tile_size_is_full_domain" : "implemented"});
  }
  for (int64_t factor : kFactors) {
    bool fullDomain = factor == kDimension;
    result.push_back({"TileReductionKParallelLinear", true,
                      "tile-reduction-k-parallel-linear", factor, !fullDomain,
                      fullDomain ? "tile_size_is_full_domain" : "implemented"});
  }
  for (int64_t factor : kFactors) {
    bool fullDomain = factor == kDimension;
    result.push_back({"TileReductionKParallelTree", true,
                      "tile-reduction-k-parallel-tree", factor, !fullDomain,
                      fullDomain ? "tile_size_is_full_domain" : "implemented"});
  }
  result.push_back({"FuseTileLocalPair", false, "fuse-tile-local-pair", 0, true,
                    "implemented"});
  result.push_back({"MaterializeNumericReduction", false,
                    "materialize-numeric-reduction", 0, true, "implemented"});
  result.push_back({"FuseReductionConsumer", false, "fuse-reduction-consumer",
                    0, true, "implemented"});
  result.push_back({"MaterializeCompletionJoin", false,
                    "materialize-completion-join", 0, true, "implemented"});
  return result;
}

struct AppliedAction {
  bool accepted = false;
  GraphState state;
  std::string reason;
  std::vector<std::string> prerequisites;
  std::vector<std::string> conflicts;
  std::vector<RegionSpec> affectedRegions;
};

static bool hasOutputTiling(const GraphState &state) {
  return state.mFactor != 0 || state.nFactor != 0;
}

static AppliedAction rejectAction(StringRef reason) {
  AppliedAction result;
  result.reason = reason.str();
  return result;
}

static AppliedAction applyAction(const GraphState &source,
                                 const ActionSpec &action) {
  AppliedAction result;
  result.state = source;
  result.prerequisites = {};
  result.conflicts = {};
  const int64_t tileSize = action.tileSize;
  if (action.kind == "TileOutputM") {
    result.prerequisites.push_back("alias_free");
    result.prerequisites.push_back("exact_static_output_region");
    result.affectedRegions.push_back(
        region("T", {0, 0}, {kDimension, kDimension}, "partition"));
    if (source.mFactor)
      return rejectAction("ALREADY_TILED");
    if (tileSize <= 1 || tileSize > kDimension)
      return rejectAction("TILE_FACTOR_INVALID");
    if (tileSize == kDimension)
      return rejectAction("TILE_FACTOR_NOOP");
    result.state.mFactor = tileSize;
    // Every output-axis tiling pass creates completion joins for the
    // resulting terminal task partitions.  Carry that semantic fact in the
    // state so an explicit completion action cannot produce a duplicate
    // candidate for the same executable witness.
    result.state.completionJoin = true;
    if (source.fusion == FusionMode::Whole)
      result.state.fusion = FusionMode::TileLocal;
  } else if (action.kind == "TileOutputN") {
    result.prerequisites.push_back("alias_free");
    result.prerequisites.push_back("exact_static_output_region");
    result.affectedRegions.push_back(
        region("T", {0, 0}, {kDimension, kDimension}, "partition"));
    if (source.nFactor)
      return rejectAction("ALREADY_TILED");
    if (tileSize <= 1 || tileSize > kDimension)
      return rejectAction("TILE_FACTOR_INVALID");
    if (tileSize == kDimension)
      return rejectAction("TILE_FACTOR_NOOP");
    result.state.nFactor = tileSize;
    result.state.completionJoin = true;
    if (source.fusion == FusionMode::Whole)
      result.state.fusion = FusionMode::TileLocal;
  } else if (action.kind == "TileReductionKSequential" ||
             action.kind == "TileReductionKParallelLinear" ||
             action.kind == "TileReductionKParallelTree") {
    result.prerequisites.push_back("exact_static_K_region");
    result.conflicts.push_back("direct_final_C_writes");
    result.affectedRegions.push_back(
        region("A", {0, 0}, {kDimension, kDimension}, "partition"));
    result.affectedRegions.push_back(
        region("B", {0, 0}, {kDimension, kDimension}, "partition"));
    if (source.kFactor)
      return rejectAction("ALREADY_TILED");
    if (tileSize <= 1 || tileSize > kDimension)
      return rejectAction("TILE_FACTOR_INVALID");
    if (tileSize == kDimension)
      return rejectAction("TILE_FACTOR_NOOP");
    result.state.kFactor = tileSize;
    if (action.kind == "TileReductionKSequential") {
      result.state.kPolicy = KPolicy::Sequential;
      result.state.reductionTopology = "none";
      result.state.numericReduction = false;
      if (source.fusion == FusionMode::Whole ||
          source.fusion == FusionMode::TileLocal)
        result.state.fusion = FusionMode::SequentialLast;
    } else {
      result.state.kPolicy = KPolicy::Parallel;
      result.state.reductionTopology =
          action.kind == "TileReductionKParallelLinear" ? "linear"
                                                        : "balanced_tree";
      result.state.numericReduction = false;
      if (source.fusion == FusionMode::Whole ||
          source.fusion == FusionMode::TileLocal) {
        result.state.fusion = FusionMode::None;
        result.state.reductionConsumerIntent = true;
      }
    }
  } else if (action.kind == "FuseProducerConsumer") {
    result.prerequisites = {"single_intermediate_user", "exact_region_match"};
    result.conflicts = {"surviving_external_T_user", "partial_consumer_direct"};
    result.affectedRegions.push_back(
        region("D", {0, 0}, {kDimension, kDimension}, "read_write"));
    if (source.fusion != FusionMode::None)
      return rejectAction("FUSION_ALREADY_PRESENT");
    if (source.kPolicy == KPolicy::Parallel)
      return rejectAction("REDUCTION_REQUIRED");
    if (hasOutputTiling(source))
      return rejectAction("FUSION_REQUIRES_UNTILED_SOURCE");
    result.state.fusion = source.kFactor            ? FusionMode::SequentialLast
                          : hasOutputTiling(source) ? FusionMode::TileLocal
                                                    : FusionMode::Whole;
  } else if (action.kind == "FuseTileLocalPair") {
    result.prerequisites = {"output_tiling_compatible", "tile_local_ready"};
    result.conflicts = {"surviving_external_T_user", "incompatible_region"};
    result.affectedRegions.push_back(
        region("D", {0, 0}, {kDimension, kDimension}, "read_write"));
    if (!hasOutputTiling(source))
      return rejectAction("FUSION_REQUIRES_TILING");
    if (source.kPolicy == KPolicy::Parallel)
      return rejectAction("REDUCTION_REQUIRED");
    if (source.fusion != FusionMode::None)
      return rejectAction("FUSION_ALREADY_PRESENT");
    result.state.fusion =
        source.kFactor ? FusionMode::SequentialLast : FusionMode::TileLocal;
  } else if (action.kind == "MaterializeNumericReduction") {
    result.prerequisites = {"parallel_k", "private_partial_buffers"};
    result.conflicts = {"completion_join_without_numeric_reduction"};
    result.affectedRegions.push_back(
        region("partial_T", {0, 0}, {kDimension, kDimension}, "reduce"));
    if (source.kPolicy != KPolicy::Parallel)
      return rejectAction("K_POLICY_CONFLICT");
    if (source.numericReduction)
      return rejectAction("NUMERIC_REDUCTION_ALREADY_PRESENT");
    result.state.numericReduction = true;
    result.state.reductionConsumerIntent = false;
  } else if (action.kind == "FuseReductionConsumer") {
    result.prerequisites = {"exact_region_and_single_user"};
    result.conflicts = {"partial_consumer_direct"};
    result.affectedRegions.push_back(
        region("D", {0, 0}, {kDimension, kDimension}, "read_write"));
    if (source.fusion != FusionMode::None)
      return rejectAction("FUSION_ALREADY_PRESENT");
    if (source.kPolicy == KPolicy::Parallel && !source.numericReduction)
      return rejectAction("REDUCTION_NOT_MATERIALIZED");
    if (source.kPolicy == KPolicy::Parallel)
      result.state.fusion = FusionMode::ReductionConsumer;
    else
      result.state.fusion = source.kFactor ? FusionMode::SequentialLast
                            : hasOutputTiling(source) ? FusionMode::TileLocal
                                                      : FusionMode::Whole;
  } else if (action.kind == "MaterializeCompletionJoin") {
    result.prerequisites = {"all_numeric_outputs_complete"};
    result.affectedRegions.push_back(
        region("D", {0, 0}, {kDimension, kDimension}, "completion"));
    if (source.completionJoin)
      return rejectAction("COMPLETION_JOIN_ALREADY_PRESENT");
    // Output tiling already materializes per-axis completion joins.  The
    // standalone completion pass is a validator/no-op when there is only one
    // terminal, so accepting this action on an untiled state would invent a
    // graph node with no corresponding executable witness.
    if (!hasOutputTiling(source))
      return rejectAction("COMPLETION_JOIN_NOOP");
    if (source.kPolicy == KPolicy::Parallel && !source.numericReduction)
      return rejectAction("INCOMPLETE_REDUCTION");
    result.state.completionJoin = true;
  } else {
    return rejectAction("ACTION_UNKNOWN");
  }
  result.accepted = true;
  return result;
}

static bool stateComplete(const GraphState &state) {
  return state.kPolicy != KPolicy::Parallel || state.numericReduction;
}

static bool isIdentityState(const GraphState &state) {
  return state.mFactor == 0 && state.nFactor == 0 && state.kFactor == 0 &&
         state.kPolicy == KPolicy::Sequential &&
         state.reductionTopology == "none" &&
         state.fusion == FusionMode::None && !state.numericReduction &&
         !state.completionJoin && !state.reductionConsumerIntent;
}

static bool validActionWitness(const ActionSpec &action) {
  if (action.pass.empty())
    return false;
  if (!action.tiled)
    return action.tileSize == 0;
  return action.tileSize == 2 || action.tileSize == 4 || action.tileSize == 8;
}

static bool isOutputTileAction(StringRef kind) {
  return kind == "TileOutputM" || kind == "TileOutputN";
}

static bool isKTileAction(StringRef kind) {
  return kind == "TileReductionKSequential" ||
         kind == "TileReductionKParallelLinear" ||
         kind == "TileReductionKParallelTree";
}

static bool isCanonicalUntiledState(const GraphState &state) {
  return state.mFactor == 0 && state.nFactor == 0 && state.kFactor == 0 &&
         state.kPolicy == KPolicy::Sequential &&
         state.reductionTopology == "none" &&
         state.fusion == FusionMode::None && !state.numericReduction &&
         !state.completionJoin && !state.reductionConsumerIntent;
}

// Action catalog entries describe the available standalone pass names.  A
// witness can use a pass only when its *preceding state* is one of the orders
// that has been replayed through the standalone passes.  Keep this metadata
// outside GraphState: changing it must never change a memoized graph ID.
static ActionSpec actionForHistory(const GraphState &source,
                                   const ActionSpec &action) {
  ActionSpec result = action;
  auto pending = [&](StringRef reason) {
    result.executable = false;
    result.materializationStatus = reason.str();
  };
  auto implemented = [&]() {
    result.executable = true;
    result.materializationStatus = "implemented";
  };

  // Full-domain M/N and K factors are represented in the semantic catalog,
  // but the standalone materializers reject them as no-op rewrites.
  if (action.tiled && action.tileSize == kDimension) {
    pending("tile_size_is_full_domain");
    return result;
  }

  if (isKTileAction(action.kind)) {
    if (!isCanonicalUntiledState(source))
      pending("pending_k_order_requires_canonical_source");
    else
      implemented();
    return result;
  }

  if (action.kind == "MaterializeNumericReduction") {
    if (source.kPolicy != KPolicy::Parallel || source.kFactor == 0 ||
        source.numericReduction || hasOutputTiling(source) ||
        source.fusion != FusionMode::None || source.reductionConsumerIntent)
      pending("pending_numeric_reduction_requires_direct_parallel_k");
    else
      implemented();
    return result;
  }

  if (action.kind == "FuseReductionConsumer") {
    bool sequentialReady = source.kPolicy == KPolicy::Sequential &&
                           source.kFactor != 0 && !hasOutputTiling(source) &&
                           source.fusion == FusionMode::None &&
                           !source.numericReduction;
    bool parallelReady =
        source.kPolicy == KPolicy::Parallel && source.kFactor != 0 &&
        source.numericReduction && !hasOutputTiling(source) &&
        source.fusion == FusionMode::None && !source.reductionConsumerIntent;
    if (sequentialReady || parallelReady)
      implemented();
    else
      pending("pending_fusion_order_requires_completed_k_reduction");
    return result;
  }

  if (isOutputTileAction(action.kind)) {
    bool directUntiled = source.kPolicy == KPolicy::Sequential &&
                         source.kFactor == 0 && !source.numericReduction &&
                         (source.fusion == FusionMode::None ||
                          source.fusion == FusionMode::Whole ||
                          source.fusion == FusionMode::TileLocal);
    bool parallelAfterReduction =
        source.kPolicy == KPolicy::Parallel && source.kFactor != 0 &&
        source.numericReduction && !source.reductionConsumerIntent &&
        (source.fusion == FusionMode::None ||
         source.fusion == FusionMode::ReductionConsumer);
    bool sequentialAfterK = source.kPolicy == KPolicy::Sequential &&
                            source.kFactor != 0 && !source.numericReduction &&
                            !source.reductionConsumerIntent &&
                            (source.fusion == FusionMode::None ||
                             source.fusion == FusionMode::SequentialLast);
    if (directUntiled || parallelAfterReduction || sequentialAfterK)
      implemented();
    else
      pending("pending_output_tiling_order_requires_supported_reduction");
    return result;
  }

  if (action.kind == "FuseProducerConsumer") {
    if (isCanonicalUntiledState(source))
      implemented();
    else
      pending("pending_fusion_order_requires_original_task_pair");
    return result;
  }

  if (action.kind == "FuseTileLocalPair") {
    if (hasOutputTiling(source) && source.kFactor == 0 &&
        source.kPolicy == KPolicy::Sequential &&
        source.fusion == FusionMode::None && !source.numericReduction)
      implemented();
    else
      pending("pending_tile_local_fusion_order_requires_output_tiling");
    return result;
  }

  if (action.kind == "MaterializeCompletionJoin")
    pending("pending_completion_join_is_materialized_by_output_tiling");
  return result;
}

static bool allActionsExecutable(ArrayRef<ActionSpec> history) {
  return llvm::all_of(
      history, [](const ActionSpec &action) { return action.executable; });
}

static llvm::json::Array stringArray(ArrayRef<std::string> values) {
  llvm::json::Array result;
  for (StringRef value : values)
    result.push_back(value.str());
  return result;
}

static std::vector<std::string> taskIds(const BuiltGraph &graph) {
  std::vector<std::string> result;
  for (const TaskSpec &task : graph.tasks)
    result.push_back(task.id);
  llvm::sort(result);
  return result;
}

static std::vector<std::string> edgeKeys(const BuiltGraph &graph) {
  std::vector<std::string> result;
  for (const EdgeSpec &edge : graph.edges)
    result.push_back(edge.source + ">" + edge.target + ":" + edge.kind + ":" +
                     edge.scope);
  llvm::sort(result);
  return result;
}

static llvm::json::Object actionJson(const ActionSpec &action) {
  llvm::json::Object result{
      {"schema", kActionSchema},
      {"kind", action.kind},
      {"action_kind", action.kind},
      {"pass", action.pass},
      {"executable", action.executable},
      {"materialization_status", action.materializationStatus},
      {"tile_factor", action.tiled ? llvm::json::Value(action.tileSize)
                                   : llvm::json::Value(nullptr)}};
  result["tile_size"] = action.tiled ? llvm::json::Value(action.tileSize)
                                     : llvm::json::Value(nullptr);
  return result;
}

static llvm::json::Array affectedRegionsJson(ArrayRef<RegionSpec> regions) {
  llvm::json::Array result;
  for (const RegionSpec &region : regions)
    result.push_back(regionJson(region));
  return result;
}

struct RewriteRecord {
  llvm::json::Object object;
};

class EnumerateJointSemanticRewritePass
    : public PassWrapper<EnumerateJointSemanticRewritePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateJointSemanticRewritePass)

  EnumerateJointSemanticRewritePass() = default;
  EnumerateJointSemanticRewritePass(
      const EnumerateJointSemanticRewritePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-joint-semantic-rewrites";
  }
  StringRef getDescription() const override {
    return "Enumerate hardware-independent semantic Taskflow rewrite states";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("canonical Taskflow function"),
      llvm::cl::init("")};
  Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("semantic rewrite manifest path"),
      llvm::cl::init("")};
  Option<int64_t> maxDepth{*this, "max-depth",
                           llvm::cl::desc("maximum rewrite DAG depth"),
                           llvm::cl::init(8)};
  Option<int64_t> maxActions{*this, "max-actions",
                             llvm::cl::desc("maximum action attempts"),
                             llvm::cl::init(50000)};
  Option<int64_t> tileFactor{
      *this, "tile-factor",
      llvm::cl::desc(
          "optional M/N/K factor filter; 0 enumerates 2,4,8; 1 is identity"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    if (outputFile.empty() || maxDepth < 0 || maxActions <= 0 ||
        (tileFactor != 0 && tileFactor != 1 && tileFactor != 2 &&
         tileFactor != 4 && tileFactor != 8)) {
      module.emitError()
          << "joint semantic enumeration requires output, nonnegative "
             "max-depth, positive max-actions, and tile-factor in {0,1,2,4,8}";
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    FailureOr<SourceContract> source =
        recognizeSource(module, *selected, error);
    if (failed(source)) {
      selected->emitError() << error;
      return signalPassFailure();
    }

    const auto enumerationStart = std::chrono::steady_clock::now();

    GraphState rootState;
    BuiltGraph root = buildGraph(*source, rootState, tileFactor);
    SmallVector<ActionSpec> catalog = actionCatalog(tileFactor);

    struct QueueItem {
      GraphState state;
      std::string id;
      int64_t depth;
      SmallVector<ActionSpec> history;
    };
    SmallVector<QueueItem> queue;
    queue.push_back({rootState, root.graphId, 0, {}});
    std::map<std::string, QueueItem> seen;
    seen.emplace(root.graphId, queue.front());
    std::map<std::string, int64_t> depthById;
    depthById[root.graphId] = 0;
    llvm::json::Array transitions;
    llvm::json::Array rejectedTransitions;
    int64_t attempted = 0;
    int64_t acceptedTransitions = 0;
    int64_t rejectedCount = 0;
    int64_t duplicateCount = 0;
    int64_t maxFrontier = 1;
    bool budgetExhausted = false;

    while (!queue.empty()) {
      QueueItem current = queue.front();
      queue.erase(queue.begin());
      if (current.depth >= maxDepth)
        continue;
      for (const ActionSpec &action : catalog) {
        if (attempted >= maxActions) {
          budgetExhausted = true;
          break;
        }
        ++attempted;
        AppliedAction applied = applyAction(current.state, action);
        ActionSpec witnessAction = actionForHistory(current.state, action);
        BuiltGraph before = buildGraph(*source, current.state, tileFactor);
        llvm::json::Object record{{"action", actionJson(witnessAction)},
                                  {"source_graph_id", current.id},
                                  {"depth", current.depth + 1}};
        if (!applied.accepted) {
          ++rejectedCount;
          record["accepted"] = false;
          record["result_graph_id"] = nullptr;
          record["reason_code"] = applied.reason;
          record["prerequisites"] = stringArray(applied.prerequisites);
          record["conflicts"] = stringArray(applied.conflicts);
          record["affected_regions"] =
              affectedRegionsJson(applied.affectedRegions);
          rejectedTransitions.push_back(std::move(record));
          continue;
        }
        ++acceptedTransitions;
        BuiltGraph after = buildGraph(*source, applied.state, tileFactor);
        bool duplicate = seen.count(after.graphId) != 0;
        record["accepted"] = true;
        record["result_graph_id"] = after.graphId;
        record["result_complete"] = stateComplete(applied.state);
        record["reason_code"] = nullptr;
        record["prerequisites"] = stringArray(applied.prerequisites);
        record["conflicts"] = stringArray(applied.conflicts);
        record["affected_regions"] =
            affectedRegionsJson(applied.affectedRegions);
        std::vector<std::string> beforeTasks = taskIds(before);
        std::vector<std::string> afterTasks = taskIds(after);
        std::vector<std::string> beforeEdges = edgeKeys(before);
        std::vector<std::string> afterEdges = edgeKeys(after);
        std::vector<std::string> createdTasks;
        std::vector<std::string> removedTasks;
        std::vector<std::string> createdEdges;
        std::vector<std::string> removedEdges;
        for (StringRef id : afterTasks)
          if (!llvm::is_contained(beforeTasks, id.str()))
            createdTasks.push_back(id.str());
        for (StringRef id : beforeTasks)
          if (!llvm::is_contained(afterTasks, id.str()))
            removedTasks.push_back(id.str());
        for (StringRef edge : afterEdges)
          if (!llvm::is_contained(beforeEdges, edge.str()))
            createdEdges.push_back(edge.str());
        for (StringRef edge : beforeEdges)
          if (!llvm::is_contained(afterEdges, edge.str()))
            removedEdges.push_back(edge.str());
        record["created_tasks"] = stringArray(createdTasks);
        record["removed_tasks"] = stringArray(removedTasks);
        record["created_edges"] = stringArray(createdEdges);
        record["removed_edges"] = stringArray(removedEdges);
        llvm::json::Object afterJson = graphJson(*source, after, tileFactor);
        if (llvm::json::Value *intermediate = afterJson.get("intermediate"))
          record["intermediate_lifetime"] = *intermediate;
        if (duplicate) {
          ++duplicateCount;
          record["duplicate_state"] = true;
        }
        transitions.push_back(std::move(record));
        int64_t nextDepth = current.depth + 1;
        SmallVector<ActionSpec> history = current.history;
        history.push_back(witnessAction);
        QueueItem next{applied.state, after.graphId, nextDepth, history};
        if (duplicate) {
          // The same semantic state can be reached by an unsupported order
          // (for example output tiling before K reduction) and by a replayed
          // supported order (K reduction before output tiling).  Keep the
          // executable witness when one is available, and revisit its
          // successors so that the preference propagates through the DAG.
          auto existing = seen.find(after.graphId);
          if (existing != seen.end() && allActionsExecutable(next.history) &&
              !allActionsExecutable(existing->second.history)) {
            existing->second = next;
            queue.push_back(next);
            maxFrontier = std::max<int64_t>(maxFrontier, queue.size());
          }
          continue;
        }
        seen.emplace(after.graphId, next);
        depthById[after.graphId] = nextDepth;
        queue.push_back(std::move(next));
        maxFrontier = std::max<int64_t>(maxFrontier, queue.size());
      }
      if (budgetExhausted)
        break;
    }
    if (budgetExhausted) {
      selected->emitError() << "rewrite enumeration exhausted max-actions="
                            << maxActions.getValue()
                            << "; refusing to publish a partial manifest";
      return signalPassFailure();
    }

    std::vector<std::pair<int64_t, std::string>> orderedIds;
    for (const auto &entry : seen)
      if (stateComplete(entry.second.state))
        orderedIds.push_back({entry.second.depth, entry.first});
    llvm::sort(orderedIds,
               [](const auto &lhs, const auto &rhs) { return lhs < rhs; });
    llvm::json::Array candidates;
    llvm::json::Array candidateIds;
    for (const auto &entry : orderedIds) {
      const QueueItem &candidateItem = seen.find(entry.second)->second;
      if ((!isIdentityState(candidateItem.state) &&
           candidateItem.history.empty()) ||
          !llvm::all_of(candidateItem.history, validActionWitness)) {
        selected->emitError()
            << "internal witness error for graph state " << entry.second
            << "; refusing to publish an unverifiable candidate";
        return signalPassFailure();
      }
      BuiltGraph graph = buildGraph(*source, candidateItem.state, tileFactor);
      llvm::json::Object candidate = graphJson(*source, graph, tileFactor);
      candidate["witness"] = witnessJson(candidateItem.history);
      candidates.push_back(std::move(candidate));
      candidateIds.push_back(graph.graphId);
    }

    llvm::json::Array actionCatalogJson;
    for (const ActionSpec &action : catalog)
      actionCatalogJson.push_back(actionJson(action));
    llvm::json::Array allRejected = std::move(rejectedTransitions);
    llvm::json::Object statistics;
    statistics["raw_action_sequences"] = attempted;
    statistics["action_attempts"] = attempted;
    statistics["accepted_transitions"] = acceptedTransitions;
    statistics["rejected_transitions"] = rejectedCount;
    statistics["unique_graph_states"] = static_cast<int64_t>(seen.size());
    statistics["construction_states"] = static_cast<int64_t>(seen.size());
    statistics["unique_candidate_graphs"] =
        static_cast<int64_t>(orderedIds.size());
    statistics["incomplete_states_removed"] =
        static_cast<int64_t>(seen.size()) -
        static_cast<int64_t>(orderedIds.size());
    statistics["duplicate_states_removed"] = duplicateCount;
    statistics["max_frontier"] = maxFrontier;
    statistics["max_depth"] = maxDepth.getValue();
    statistics["max_actions"] = maxActions.getValue();
    statistics["budget_exhausted"] = false;
    const auto enumerationEnd = std::chrono::steady_clock::now();
    statistics["enumeration_time_seconds"] =
        std::chrono::duration<double>(enumerationEnd - enumerationStart)
            .count();

    llvm::json::Object manifest;
    manifest["schema"] = kDagSchema;
    manifest["complete"] = true;
    manifest["source_module_sha256"] = source->moduleSha256;
    manifest["source_module_sha256_kind"] = "normalized_mlir_print";
    manifest["source_module_sha256_exact_input_bytes"] = false;
    manifest["canonical_source_id"] = source->moduleSha256;
    manifest["function"] =
        source->function->getAttrOfType<StringAttr>("sym_name")
            .getValue()
            .str();
    manifest["dimensions"] = llvm::json::Object{
        {"M", kDimension}, {"N", kDimension}, {"K", kDimension}};
    manifest["dtype"] = "i32";
    manifest["layout"] = "row_major_identity";
    manifest["arithmetic_semantics"] = kArithmeticSemantics;
    manifest["tile_factor"] = tileFactor.getValue();
    manifest["tile_factor_filter"] = tileFactor.getValue();
    llvm::json::Array enumeratedTileFactors;
    if (tileFactor == 0) {
      enumeratedTileFactors.push_back(2);
      enumeratedTileFactors.push_back(4);
      enumeratedTileFactors.push_back(8);
    } else {
      enumeratedTileFactors.push_back(tileFactor.getValue());
    }
    manifest["enumerated_tile_factors"] = std::move(enumeratedTileFactors);
    manifest["statistics"] = std::move(statistics);
    manifest["action_catalog"] = std::move(actionCatalogJson);
    manifest["candidates"] = std::move(candidates);
    manifest["candidate_graph_ids"] = std::move(candidateIds);
    manifest["transitions"] = std::move(transitions);
    manifest["rejections"] = std::move(allRejected);
    std::string writeError;
    if (!writeAtomically(
            outputFile,
            [&](llvm::raw_ostream &os) {
              os << llvm::json::Value(std::move(manifest)) << "\n";
              return true;
            },
            writeError)) {
      selected->emitError() << writeError;
      return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createEnumerateJointSemanticRewritePass() {
  return std::make_unique<EnumerateJointSemanticRewritePass>();
}

} // namespace mlir::amoeba::neura
