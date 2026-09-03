//===- AnalyticalTaskDSEPasses.cpp - mapper-free task DSE ---------------===//
//
// The production task-level DSE lives in Amoeba because Taskflow owns the
// transformation legality and replay semantics.  The flow is deliberately
// split into three passes:
//
//   enumerate:   freeze every hard-valid candidate without looking at scores;
//   score:       score the complete frozen manifest, then emit top-k;
//   materialize: apply exactly one selected candidate, without running mapper.
//
// This first executable slice searches rectangular task shapes only.  The
// manifest fixes the other axes to identity choices and names its score a
// compute bottleneck rather than a pipeline interval.  Fusion, partitioning,
// placement, temporal order, and communication can extend the candidate
// record without weakening the enumerate-before-score boundary.
//
// End-to-end ownership and data flow
// ----------------------------------
//
//   Taskflow IR + architecture spec
//                |
//                v
//   EnumerateAnalyticalTaskCandidatesPass
//                |  complete, deterministic candidate JSONL
//                v
//   ML predictor output (task-shape cost catalogue)
//                |
//                v
//   ScoreAnalyticalTaskCandidatesPass
//                |  one score record per candidate + top-k footer
//                v
//   external DSE driver
//                |  one invocation for each shortlisted candidate
//                v
//   MaterializeAnalyticalTaskCandidatePass -> unchanged heuristic mapper
//
// The passes communicate through files on purpose.  Enumeration can finish
// before the ML service is involved, the exact search space can be archived,
// and a shortlist can be replayed without silently regenerating a different
// candidate set.  Each reader therefore treats its input as untrusted: schema,
// task facts, architecture fingerprint, record order, record count, IDs, and
// footer digest are checked before results are accepted.
//
// Terminology used below
// ----------------------
//
// * physical shape: rectangle of CGRA instances assigned to one task, e.g.
//   1x2 CGRAs.  This becomes taskflow.task's cgra_shape/cgra_count.
// * mapper shape: the corresponding rectangle in the coordinate system seen
//   by the single-task mapper.  If one CGRA is 4x4 tiles, physical 1x2 becomes
//   mapper 4x8.
// * task body ID: stable hash of the task computation after removing DSE and
//   profiling attributes.  It prevents a stale ML prediction from being used
//   after the computation changes.
// * candidate: one ordered physical-shape choice per Taskflow task.
// * cost query: (task name, task body ID, mapper shape).  Many candidates use
//   the same query, so scoring memoizes it once for the whole pass invocation.
//
// Three non-negotiable ordering invariants implement the intended DSE:
//
// 1. Enumeration never sees ML scores and never truncates the search space.
//    max-candidates is a safety limit that fails atomically; it is not pruning.
// 2. Scoring emits a record for every frozen candidate before sorting and
//    selecting top-k.  Unsupported task-shape pairs invalidate a candidate but
//    do not make it disappear from the score stream.
// 3. Materialization validates the complete manifest before changing the IR.
//    It only writes the selected shape; it never invokes the real mapper.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates-v1";
constexpr StringLiteral kCostSchema = "amoeba-task-shape-cost-v1";
constexpr StringLiteral kScoreSchema = "amoeba-analytical-task-scores-v1";
constexpr StringLiteral kShapePolicy = "rectangles-v1";
constexpr StringLiteral kSearchScope = "shape-only-v1";
constexpr StringLiteral kCandidateIdentity = "shape-candidate-sha256-v1";
constexpr StringLiteral kScoreModel = "shape-only-compute-bottleneck-v1";

//===----------------------------------------------------------------------===//
// In-memory records shared by the three passes
//===----------------------------------------------------------------------===//

// One shape has two coordinate systems.  rows/cols count physical CGRAs;
// mapperRows/mapperCols count the tiles presented to Neura's mapper.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;
  std::string mapperShapeId;

  int64_t cgraCount() const { return rows * cols; }
  std::string irAttr() const {
    return std::to_string(rows) + "x" + std::to_string(cols);
  }
};

// Immutable facts extracted from the current IR.  A manifest is reusable only
// while all three values still match at the same task position.
struct TaskFact {
  TaskflowTaskOp op;
  std::string name;
  std::string bodyId;
  int64_t tripCount = 1;
};

// A candidate repeats task facts next to the chosen shape.  The redundancy is
// intentional: it makes a JSONL record self-describing and lets the reader
// catch a manifest replayed on a different Taskflow program.
struct TaskShapeChoice {
  std::string task;
  std::string bodyId;
  int64_t tripCount = 1;
  RectShape shape;
};

// Task order follows func.walk order and is part of candidate identity.
struct Candidate {
  std::string id;
  SmallVector<TaskShapeChoice> choices;
};

// Header values define the finite candidate space.  The footer proves that
// every point in that space was emitted exactly once and in canonical order.
struct ManifestHeader {
  std::string function;
  std::string architectureFingerprint;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  int64_t perCgraRows = 0;
  int64_t perCgraCols = 0;
  int64_t maxCgrasPerTask = 0;
};

struct ManifestFooter {
  uint64_t candidateCount = 0;
  std::string candidateIdsSha256;
};

// ML output for one query.  A supported pair models execution as
//
//   startupCycles + predictedII * (tripCount - 1).
//
// An unsupported pair remains a valid catalogue entry so the score file can
// explain why candidates using it were rejected.
struct BodyShapeCost {
  double predictedII = 0.0;
  double startupCycles = 0.0;
  bool supported = false;
};

// task name distinguishes separate task instances; body ID protects semantic
// identity; mapper shape distinguishes the spatial resource presented to ML.
using CostKey = std::tuple<std::string, std::string, std::string>;

struct RankedCandidate {
  std::string id;
  double score = 0.0;
};

//===----------------------------------------------------------------------===//
// Stable identities and Taskflow facts
//===----------------------------------------------------------------------===//

static std::string sha256(StringRef text) {
  llvm::SHA256 hasher;
  hasher.update(text);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static void updateIdDigest(llvm::SHA256 &hasher, StringRef candidateId) {
  hasher.update(candidateId);
  hasher.update("\n");
}

// Bind every manifest and task-shape cost catalogue to the exact architecture
// specification selected by the Amoeba driver.  Hashing the file is
// intentionally conservative: even semantically equivalent edits invalidate
// stale scores instead of silently reusing them for a changed machine model.
static FailureOr<std::string>
currentArchitectureFingerprint(std::string &error) {
  StringRef path = mlir::amoeba::getNeuraArchitectureSpecFile();
  if (path.empty()) {
    error = "analytical task DSE requires --architecture-spec";
    return failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot fingerprint architecture specification " + path.str() +
            ": " + buffer.getError().message();
    return failure();
  }
  return "sha256:" + sha256((*buffer)->getBuffer());
}

// Matches frontend_model.shapes: the mapper ID names the exact tile mask, not
// just its bounding box.  For rectangles the mask is the whole tile array.
static std::string mapperShapeId(int64_t rows, int64_t cols) {
  std::string canonical = "[";
  bool first = true;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      if (!first)
        canonical += ",";
      first = false;
      canonical += "{\"col\":" + std::to_string(col) +
                   ",\"row\":" + std::to_string(row) + "}";
    }
  }
  canonical += "]";
  return std::to_string(rows) + "x" + std::to_string(cols) + "-" +
         sha256(canonical).substr(0, 10);
}

static std::string taskBodyId(TaskflowTaskOp task) {
  if (auto explicitId = task->getAttrOfType<StringAttr>("analytical_body_id"))
    if (!explicitId.getValue().empty())
      return explicitId.getValue().str();

  // Hash the computation/interface, not transient decisions or measurements.
  // Otherwise materializing cgra_shape (or merely profiling the same body)
  // changes the cache key and prevents reuse of the same (task, shape) query.
  Operation *clone = task->clone();
  auto destroyClone = llvm::make_scope_exit([&] { clone->destroy(); });
  clone->setAttr("task_name",
                 StringAttr::get(task.getContext(), "__analytical_body__"));
  for (StringRef attr :
       {"analytical_body_id", "trip_count", "cgra_count", "cgra_shape",
        "compiled_ii", "profile_info", "task_orchestration_info", "replicas",
        "tiling", "est_latency"})
    clone->removeAttr(attr);
  std::string printed;
  llvm::raw_string_ostream os(printed);
  OpPrintingFlags flags;
  flags.printGenericOpForm().useLocalScope();
  clone->print(os, flags);
  os.flush();
  return "sha256:" + sha256(printed);
}

static FailureOr<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return failure();
}

// Mirrors the existing ResourceAware convention: multiply counters along a
// root chain and take the maximum over independent roots.  Dynamic bounds must
// be made explicit with a trip_count attribute rather than silently guessed.
static FailureOr<int64_t> taskTripCount(TaskflowTaskOp task,
                                        std::string &error) {
  if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
    if (attr.getInt() <= 0) {
      error =
          "task " + task.getTaskName().str() + " has non-positive trip_count";
      return failure();
    }
    return attr.getInt();
  }

  if (!task.getBody().hasOneBlock()) {
    error = "task " + task.getTaskName().str() +
            " must have one block or an explicit trip_count";
    return failure();
  }

  SmallVector<TaskflowCounterOp> counters;
  for (Operation &op : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&op))
      counters.push_back(counter);
  if (counters.empty())
    return int64_t{1};

  SmallVector<TaskflowCounterOp> roots;
  DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex())
      children[parent].push_back(counter);
    else
      roots.push_back(counter);
  }
  if (roots.empty()) {
    error = "task " + task.getTaskName().str() +
            " has counters but no root counter";
    return failure();
  }

  auto counterTrip = [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    FailureOr<int64_t> lower = constantIndex(counter.getLowerBound());
    FailureOr<int64_t> upper = constantIndex(counter.getUpperBound());
    FailureOr<int64_t> step = constantIndex(counter.getStep());
    if (failed(lower) || failed(upper) || failed(step) || *step <= 0 ||
        *upper <= *lower)
      return failure();
    return (*upper - *lower + *step - 1) / *step;
  };

  int64_t result = 1;
  for (TaskflowCounterOp root : roots) {
    int64_t product = 1;
    SmallVector<TaskflowCounterOp> worklist{root};
    while (!worklist.empty()) {
      TaskflowCounterOp counter = worklist.pop_back_val();
      FailureOr<int64_t> count = counterTrip(counter);
      if (failed(count) ||
          product > std::numeric_limits<int64_t>::max() / *count) {
        error = "task " + task.getTaskName().str() +
                " has dynamic, invalid, or overflowing counter bounds; add "
                "an explicit trip_count";
        return failure();
      }
      product *= *count;
      auto found = children.find(counter.getCounterIndex());
      if (found != children.end())
        worklist.append(found->second.begin(), found->second.end());
    }
    result = std::max(result, product);
  }
  return result;
}

static FailureOr<SmallVector<TaskFact>> collectTaskFacts(func::FuncOp func,
                                                         std::string &error) {
  // The ordered vector collected here is the task axis of the Cartesian
  // product.  Duplicate names would make cost keys and replay ambiguous, so
  // reject them before either a manifest or score file is written.
  SmallVector<TaskFact> tasks;
  llvm::StringSet<> names;
  WalkResult walkResult = func.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    FailureOr<int64_t> tripCount = taskTripCount(task, error);
    if (failed(tripCount))
      return WalkResult::interrupt();
    tasks.push_back({task, std::move(name), taskBodyId(task), *tripCount});
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  if (tasks.empty()) {
    error = "function contains no taskflow.task operations";
    return failure();
  }
  return tasks;
}

//===----------------------------------------------------------------------===//
// Shape space and canonical candidate encoding
//===----------------------------------------------------------------------===//

// Enumerate all factor pairs for each physical CGRA count.  Only rectangles
// fitting the multi-CGRA grid are legal.  Shapes are ordered first by resource
// count, then from the most square to the most elongated; this order is frozen
// into manifest indices and must be reproduced by the reader.
static SmallVector<RectShape>
enumerateRectShapes(int64_t gridRows, int64_t gridCols, int64_t perCgraRows,
                    int64_t perCgraCols, int64_t maxCgrasPerTask) {
  SmallVector<RectShape> result;
  const int64_t maxCount = std::min(maxCgrasPerTask, gridRows * gridCols);
  for (int64_t count = 1; count <= maxCount; ++count) {
    SmallVector<RectShape> atCount;
    for (int64_t rows = 1; rows <= count; ++rows) {
      if (count % rows != 0)
        continue;
      int64_t cols = count / rows;
      if (rows > gridRows || cols > gridCols)
        continue;
      int64_t mapperRows = rows * perCgraRows;
      int64_t mapperCols = cols * perCgraCols;
      atCount.push_back({rows, cols, mapperRows, mapperCols,
                         mapperShapeId(mapperRows, mapperCols)});
    }
    llvm::sort(atCount, [](const RectShape &lhs, const RectShape &rhs) {
      return std::make_tuple(std::abs(lhs.rows - lhs.cols), lhs.rows,
                             lhs.cols) <
             std::make_tuple(std::abs(rhs.rows - rhs.cols), rhs.rows, rhs.cols);
    });
    result.append(atCount.begin(), atCount.end());
  }
  return result;
}

static void appendIdentityField(std::string &out, StringRef key,
                                StringRef value) {
  // Length-prefix every value so concatenation cannot alias, e.g. ("ab", "c")
  // and ("a", "bc") produce different byte streams before hashing.
  out += key.str();
  out += ":";
  out += std::to_string(value.size());
  out += ":";
  out += value.str();
  out += "\n";
}

static std::string candidateId(StringRef function,
                               StringRef architectureFingerprint,
                               ArrayRef<TaskShapeChoice> choices) {
  // Do not hash the pretty-printed JSON: JSON field order/whitespace are an
  // interchange detail.  This explicit canonical stream defines exactly which
  // semantic changes invalidate a candidate ID.
  std::string identity;
  appendIdentityField(identity, "schema", kCandidateIdentity);
  appendIdentityField(identity, "shape_policy", kShapePolicy);
  appendIdentityField(identity, "function", function);
  appendIdentityField(identity, "architecture", architectureFingerprint);
  for (const TaskShapeChoice &choice : choices) {
    appendIdentityField(identity, "task", choice.task);
    appendIdentityField(identity, "body_id", choice.bodyId);
    appendIdentityField(identity, "trip_count",
                        std::to_string(choice.tripCount));
    appendIdentityField(identity, "physical_shape", choice.shape.irAttr());
    appendIdentityField(identity, "mapper_shape", choice.shape.mapperShapeId);
  }
  return sha256(identity);
}

static llvm::json::Object shapeJson(const RectShape &shape) {
  llvm::json::Object object;
  object["kind"] = "rect";
  object["rows"] = shape.rows;
  object["cols"] = shape.cols;
  object["cgra_count"] = shape.cgraCount();
  object["cgra_shape"] = shape.irAttr();
  object["mapper_tile_rows"] = shape.mapperRows;
  object["mapper_tile_cols"] = shape.mapperCols;
  object["mapper_shape_id"] = shape.mapperShapeId;
  return object;
}

static llvm::json::Object candidateJson(const Candidate &candidate) {
  llvm::json::Array choices;
  for (const TaskShapeChoice &choice : candidate.choices) {
    llvm::json::Object record;
    record["task"] = choice.task;
    record["body_id"] = choice.bodyId;
    record["trip_count"] = choice.tripCount;
    record["shape"] = shapeJson(choice.shape);
    choices.push_back(std::move(record));
  }
  llvm::json::Object record;
  record["record_type"] = "candidate";
  record["schema_version"] = kCandidateSchema.str();
  record["candidate_id"] = candidate.id;
  record["task_shapes"] = std::move(choices);
  return record;
}

static void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

static bool writeAtomically(StringRef output,
                            llvm::function_ref<bool(llvm::raw_ostream &)> body,
                            std::string &error) {
  // A partial candidate or score file is dangerous because a downstream
  // driver could mistake it for a pruned search space.  Always write a sibling
  // temporary file and rename it only after the body and stream both succeed.
  if (output.empty()) {
    error = "output path is required";
    return false;
  }
  SmallString<256> pattern(output);
  pattern += ".tmp-%%%%%%";
  SmallString<256> temporary;
  int descriptor = -1;
  std::error_code ec =
      llvm::sys::fs::createUniqueFile(pattern, descriptor, temporary);
  if (ec) {
    error = "cannot create temporary output " + temporary.str().str() + ": " +
            ec.message();
    return false;
  }

  bool ok = false;
  {
    llvm::raw_fd_ostream os(descriptor, /*shouldClose=*/true);
    ok = body(os);
    os.flush();
    if (os.has_error()) {
      error = "failed while writing " + output.str();
      ok = false;
    }
  }
  if (!ok) {
    llvm::sys::fs::remove(temporary);
    return false;
  }
  ec = llvm::sys::fs::rename(temporary, output);
  if (ec) {
    error = "cannot publish " + output.str() + ": " + ec.message();
    llvm::sys::fs::remove(temporary);
    return false;
  }
  return true;
}

static SmallString<256> normalizedPath(StringRef path) {
  SmallString<256> result(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(result))
    return SmallString<256>(path);
  llvm::sys::path::remove_dots(result, /*remove_dot_dot=*/true);
  return result;
}

static bool samePath(StringRef lhs, StringRef rhs) {
  bool equivalent = false;
  if (!llvm::sys::fs::equivalent(lhs, rhs, equivalent) && equivalent)
    return true;
  return normalizedPath(lhs) == normalizedPath(rhs);
}

static FailureOr<func::FuncOp>
selectTaskFunction(ModuleOp module, StringRef requested, std::string &error) {
  // Module-level passes need an unambiguous Taskflow function.  Large modules
  // can opt in explicitly; the common single-function case stays convenient.
  if (!requested.empty()) {
    auto function = module.lookupSymbol<func::FuncOp>(requested);
    if (!function) {
      error = "requested function " + requested.str() + " does not exist";
      return failure();
    }
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (!hasTask) {
      error = "requested function " + requested.str() +
              " contains no taskflow.task operations";
      return failure();
    }
    return function;
  }

  SmallVector<func::FuncOp> taskFunctions;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask)
      taskFunctions.push_back(function);
  }
  if (taskFunctions.size() != 1) {
    error = "expected exactly one function containing Taskflow tasks; use the "
            "function option when the module contains more than one";
    return failure();
  }
  return taskFunctions.front();
}

//===----------------------------------------------------------------------===//
// Candidate-manifest validation and streaming
//===----------------------------------------------------------------------===//

// Manifest parsing is deliberately stricter than ordinary configuration-file
// parsing.  The file is the boundary between exhaustive enumeration and
// scoring/materialization; accepting an omitted or altered record here would
// silently turn exhaustive DSE into heuristic pruning.
static std::optional<StringRef> requiredString(const llvm::json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value)
    error = "missing or invalid string field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> requiredInteger(const llvm::json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> checkedPositiveProduct(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::nullopt;
  return lhs * rhs;
}

static bool parseHeader(const llvm::json::Object &object,
                        ManifestHeader &header, std::string &error) {
  // The version strings give later extensions (fusion, tiling, temporal, ...)
  // an explicit migration point instead of reinterpreting an old manifest.
  std::optional<StringRef> schema =
      requiredString(object, "schema_version", error);
  std::optional<StringRef> function = requiredString(object, "function", error);
  std::optional<StringRef> scope =
      requiredString(object, "search_scope", error);
  std::optional<StringRef> policy =
      requiredString(object, "shape_policy", error);
  std::optional<StringRef> identity =
      requiredString(object, "candidate_identity", error);
  const llvm::json::Object *architecture = object.getObject("architecture");
  if (!schema || !function || !scope || !policy || !identity || !architecture)
    return false;
  if (*schema != kCandidateSchema || *scope != kSearchScope ||
      *policy != kShapePolicy || *identity != kCandidateIdentity) {
    error = "unsupported candidate manifest contract";
    return false;
  }
  auto gridRows = requiredInteger(*architecture, "grid_rows", error);
  auto gridCols = requiredInteger(*architecture, "grid_cols", error);
  auto perRows = requiredInteger(*architecture, "per_cgra_tile_rows", error);
  auto perCols = requiredInteger(*architecture, "per_cgra_tile_cols", error);
  auto fingerprint = requiredString(*architecture, "spec_fingerprint", error);
  auto maxCgras = requiredInteger(object, "max_cgras_per_task", error);
  if (!gridRows || !gridCols || !perRows || !perCols || !fingerprint ||
      !maxCgras)
    return false;
  header = {function->str(), fingerprint->str(), *gridRows, *gridCols,
            *perRows,        *perCols,           *maxCgras};
  if (header.gridRows <= 0 || header.gridCols <= 0 || header.perCgraRows <= 0 ||
      header.perCgraCols <= 0 || header.maxCgrasPerTask <= 0) {
    error = "candidate manifest dimensions must be positive";
    return false;
  }
  StringRef fingerprintValue(header.architectureFingerprint);
  if (!fingerprintValue.consume_front("sha256:") ||
      fingerprintValue.size() != 64 ||
      llvm::any_of(fingerprintValue,
                   [](char value) { return !llvm::isHexDigit(value); })) {
    error = "candidate manifest architecture fingerprint is invalid";
    return false;
  }
  return true;
}

static bool validateHeaderTasks(const llvm::json::Object &object,
                                ArrayRef<TaskFact> tasks, std::string &error) {
  const llvm::json::Array *records = object.getArray("tasks");
  if (!records || records->size() != tasks.size()) {
    error = "candidate manifest header task list does not match current IR";
    return false;
  }
  for (auto [index, value] : llvm::enumerate(*records)) {
    const llvm::json::Object *record = value.getAsObject();
    if (!record) {
      error = "candidate manifest header task is not an object";
      return false;
    }
    auto name = requiredString(*record, "task", error);
    auto bodyId = requiredString(*record, "body_id", error);
    auto tripCount = requiredInteger(*record, "trip_count", error);
    if (!name || !bodyId || !tripCount)
      return false;
    if (*name != tasks[index].name || *bodyId != tasks[index].bodyId ||
        *tripCount != tasks[index].tripCount) {
      error = "candidate manifest header task facts do not match current IR";
      return false;
    }
  }
  return true;
}

static bool parseShape(const llvm::json::Object &object, RectShape &shape,
                       std::string &error) {
  auto kind = requiredString(object, "kind", error);
  auto rows = requiredInteger(object, "rows", error);
  auto cols = requiredInteger(object, "cols", error);
  auto count = requiredInteger(object, "cgra_count", error);
  auto irShape = requiredString(object, "cgra_shape", error);
  auto mapperRows = requiredInteger(object, "mapper_tile_rows", error);
  auto mapperCols = requiredInteger(object, "mapper_tile_cols", error);
  auto mapperId = requiredString(object, "mapper_shape_id", error);
  if (!kind || !rows || !cols || !count || !irShape || !mapperRows ||
      !mapperCols || !mapperId)
    return false;
  std::optional<int64_t> computedCount = checkedPositiveProduct(*rows, *cols);
  if (*kind != "rect" || !computedCount || *count != *computedCount ||
      *mapperRows <= 0 || *mapperCols <= 0) {
    error = "candidate contains a non-rectangular or invalid shape";
    return false;
  }
  shape = {*rows, *cols, *mapperRows, *mapperCols, mapperId->str()};
  if (*irShape != shape.irAttr()) {
    error = "candidate shape identifier does not match its dimensions";
    return false;
  }
  return true;
}

static bool sameShape(const RectShape &lhs, const RectShape &rhs) {
  return std::tie(lhs.rows, lhs.cols, lhs.mapperRows, lhs.mapperCols,
                  lhs.mapperShapeId) == std::tie(rhs.rows, rhs.cols,
                                                 rhs.mapperRows, rhs.mapperCols,
                                                 rhs.mapperShapeId);
}

static bool parseCandidate(const llvm::json::Object &object,
                           const ManifestHeader &header,
                           ArrayRef<TaskFact> tasks,
                           ArrayRef<RectShape> legalShapes,
                           Candidate &candidate, std::string &error) {
  // Validation proceeds from cheap structural checks to semantic checks:
  // current task facts, dimension arithmetic, membership in the declared
  // shape family, and finally recomputation of the stable candidate ID.
  auto schema = requiredString(object, "schema_version", error);
  auto id = requiredString(object, "candidate_id", error);
  const llvm::json::Array *records = object.getArray("task_shapes");
  if (!schema || !id || !records)
    return false;
  if (*schema != kCandidateSchema || records->size() != tasks.size()) {
    error = "candidate schema or task count does not match its manifest";
    return false;
  }

  candidate.choices.clear();
  for (auto [index, value] : llvm::enumerate(*records)) {
    const llvm::json::Object *record = value.getAsObject();
    if (!record) {
      error = "task_shapes entry is not an object";
      return false;
    }
    auto taskName = requiredString(*record, "task", error);
    auto bodyId = requiredString(*record, "body_id", error);
    auto tripCount = requiredInteger(*record, "trip_count", error);
    const llvm::json::Object *shapeObject = record->getObject("shape");
    if (!taskName || !bodyId || !tripCount || !shapeObject)
      return false;
    const TaskFact &task = tasks[index];
    if (*taskName != task.name || *bodyId != task.bodyId ||
        *tripCount != task.tripCount) {
      error = "candidate task facts do not match the current IR";
      return false;
    }
    RectShape shape;
    if (!parseShape(*shapeObject, shape, error))
      return false;
    std::optional<int64_t> expectedMapperRows =
        checkedPositiveProduct(shape.rows, header.perCgraRows);
    std::optional<int64_t> expectedMapperCols =
        checkedPositiveProduct(shape.cols, header.perCgraCols);
    if (!expectedMapperRows || !expectedMapperCols ||
        shape.mapperRows != *expectedMapperRows ||
        shape.mapperCols != *expectedMapperCols ||
        llvm::none_of(legalShapes, [&](const RectShape &item) {
          return sameShape(item, shape);
        })) {
      error = "candidate shape is outside the declared rectangular space";
      return false;
    }
    candidate.choices.push_back(
        {task.name, task.bodyId, task.tripCount, std::move(shape)});
  }
  candidate.id = candidateId(header.function, header.architectureFingerprint,
                             candidate.choices);
  if (*id != candidate.id) {
    error = "candidate_id does not match canonical candidate content";
    return false;
  }
  return true;
}

static FailureOr<uint64_t> expectedCandidateCount(size_t shapeCount,
                                                  size_t taskCount,
                                                  std::string &error) {
  if (shapeCount == 0) {
    error = "candidate manifest declares an empty shape space";
    return failure();
  }
  uint64_t result = 1;
  for (size_t ignored = 0; ignored < taskCount; ++ignored) {
    if (result > std::numeric_limits<uint64_t>::max() / shapeCount) {
      error = "declared candidate count overflows uint64";
      return failure();
    }
    result *= shapeCount;
  }
  return result;
}

// Enumeration order is part of the frozen contract. Checking the exact
// expected record at each index detects omissions, duplicates, and reorderings
// even if the footer digest is recomputed after editing the JSONL.
static bool validateCandidateAtIndex(uint64_t index, ArrayRef<TaskFact> tasks,
                                     ArrayRef<RectShape> shapes,
                                     const Candidate &candidate,
                                     std::string &error) {
  SmallVector<size_t> shapeIndices(tasks.size());
  uint64_t remainder = index;
  for (size_t reverse = tasks.size(); reverse > 0; --reverse) {
    shapeIndices[reverse - 1] = remainder % shapes.size();
    remainder /= shapes.size();
  }
  if (remainder != 0 || candidate.choices.size() != tasks.size()) {
    error = "candidate manifest contains more records than its declared space";
    return false;
  }
  for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
    if (!sameShape(candidate.choices[taskIndex].shape,
                   shapes[shapeIndices[taskIndex]])) {
      error = "candidate manifest is incomplete, duplicated, or out of order";
      return false;
    }
  }
  return true;
}

using CandidateConsumer =
    llvm::function_ref<bool(uint64_t, const Candidate &, std::string &)>;

static bool architectureMatches(const ManifestHeader &header,
                                const neura::Architecture &architecture,
                                StringRef currentFingerprint,
                                std::string &error) {
  if (header.gridRows != architecture.getMultiCgraRows() ||
      header.gridCols != architecture.getMultiCgraColumns() ||
      header.perCgraRows != architecture.getPerCgraRows() ||
      header.perCgraCols != architecture.getPerCgraColumns() ||
      header.architectureFingerprint != currentFingerprint) {
    error = "candidate manifest architecture does not match current Neura "
            "architecture";
    return false;
  }
  return true;
}

static bool readCandidateManifest(StringRef path, ArrayRef<TaskFact> tasks,
                                  StringRef expectedFunction,
                                  const neura::Architecture &architecture,
                                  StringRef architectureFingerprint,
                                  CandidateConsumer consume,
                                  ManifestHeader &header,
                                  ManifestFooter &footer, std::string &error) {
  // Stream candidates rather than retaining the Cartesian product in memory.
  // The consumer can score a record or remember a requested record, while this
  // routine owns all completeness and provenance checks common to both uses.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read candidate manifest " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }

  bool sawHeader = false;
  bool sawFooter = false;
  uint64_t count = 0;
  uint64_t declaredSpaceCount = 0;
  SmallVector<RectShape> legalShapes;
  llvm::SHA256 idsDigest;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      error = "invalid candidate JSONL: " + llvm::toString(parsed.takeError());
      return false;
    }
    llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "candidate JSONL record is not an object";
      return false;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      error = "candidate JSONL record has no record_type";
      return false;
    }
    if (*recordType == "header") {
      // The header must be first.  Reconstructing legalShapes from it gives us
      // both the expected product size and the canonical mixed-radix ordering.
      if (sawHeader || count != 0 || sawFooter ||
          !parseHeader(*object, header, error) ||
          !validateHeaderTasks(*object, tasks, error)) {
        if (error.empty())
          error = "candidate manifest header is misplaced or duplicated";
        return false;
      }
      if (header.function != expectedFunction) {
        error = "candidate manifest function does not match current IR";
        return false;
      }
      if (!architectureMatches(header, architecture, architectureFingerprint,
                               error))
        return false;
      legalShapes = enumerateRectShapes(header.gridRows, header.gridCols,
                                        header.perCgraRows, header.perCgraCols,
                                        header.maxCgrasPerTask);
      FailureOr<uint64_t> computed =
          expectedCandidateCount(legalShapes.size(), tasks.size(), error);
      if (failed(computed))
        return false;
      declaredSpaceCount = *computed;
      sawHeader = true;
      continue;
    }
    if (*recordType == "candidate") {
      // validateCandidateAtIndex is stronger than checking IDs alone: an
      // attacker/editor cannot remove or reorder records and then merely
      // recompute the footer digest.
      if (!sawHeader || sawFooter) {
        error = "candidate record is outside header/footer";
        return false;
      }
      Candidate candidate;
      if (!parseCandidate(*object, header, tasks, legalShapes, candidate,
                          error) ||
          !validateCandidateAtIndex(count, tasks, legalShapes, candidate,
                                    error) ||
          !consume(count, candidate, error))
        return false;
      updateIdDigest(idsDigest, candidate.id);
      ++count;
      continue;
    }
    if (*recordType == "footer") {
      // The footer closes the stream only when the declared count, mathematically
      // expected product size, and digest of all ordered IDs agree.
      auto schema = requiredString(*object, "schema_version", error);
      auto expectedCount = requiredInteger(*object, "candidate_count", error);
      auto digest = requiredString(*object, "candidate_ids_sha256", error);
      if (!sawHeader || sawFooter || !schema || !expectedCount || !digest ||
          *schema != kCandidateSchema || *expectedCount < 0) {
        if (error.empty())
          error = "invalid candidate manifest footer";
        return false;
      }
      std::string actualDigest =
          llvm::toHex(idsDigest.final(), /*LowerCase=*/true);
      if (static_cast<uint64_t>(*expectedCount) != count ||
          count != declaredSpaceCount || *digest != actualDigest) {
        error = "candidate manifest count/digest mismatch";
        return false;
      }
      footer = {count, actualDigest};
      sawFooter = true;
      continue;
    }
    error = "unknown candidate JSONL record_type " + recordType->str();
    return false;
  }
  if (!sawHeader || !sawFooter) {
    error = "candidate manifest is incomplete";
    return false;
  }
  return true;
}

// Adapter between the scorer and the ML agent's output file.
//
// catalog_ is the validated, immutable set of predictions supplied by ML.
// cache_ is the set actually requested while walking candidates.  Keeping the
// two maps separate lets us report real hit/miss statistics and, later, replace
// catalog lookup with an online model call without changing candidate scoring.
// The model namespace is catalog-wide, while CostKey contains the task/body/
// shape identity.  The architecture fingerprint and catalogue SHA are recorded
// in the score output so results from different model/machine versions cannot
// be confused.
class BodyShapeCostCache {
public:
  bool load(StringRef path, StringRef expectedFunction,
            StringRef expectedArchitectureFingerprint, std::string &error) {
    // Validate the whole catalogue eagerly.  A duplicate or malformed entry is
    // a producer error even if no current candidate happens to query it.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      error = "cannot read cost catalogue " + path.str() + ": " +
              buffer.getError().message();
      return false;
    }
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      error =
          "invalid cost catalogue JSON: " + llvm::toString(parsed.takeError());
      return false;
    }
    catalogSha256_ = sha256((*buffer)->getBuffer());
    llvm::json::Object *root = parsed->getAsObject();
    if (!root) {
      error = "cost catalogue must be a JSON object";
      return false;
    }
    auto schema = requiredString(*root, "schema_version", error);
    auto function = requiredString(*root, "function", error);
    auto modelNamespace = requiredString(*root, "namespace", error);
    auto architectureFingerprint =
        requiredString(*root, "architecture_fingerprint", error);
    llvm::json::Array *entries = root->getArray("entries");
    if (!schema || !function || !modelNamespace || !architectureFingerprint ||
        !entries)
      return false;
    if (*schema != kCostSchema || *function != expectedFunction ||
        modelNamespace->empty() ||
        *architectureFingerprint != expectedArchitectureFingerprint) {
      error = "cost catalogue schema/function/namespace/architecture mismatch";
      return false;
    }
    namespace_ = modelNamespace->str();

    for (llvm::json::Value &value : *entries) {
      llvm::json::Object *entry = value.getAsObject();
      if (!entry) {
        error = "cost entry is not an object";
        return false;
      }
      auto task = requiredString(*entry, "task", error);
      auto bodyId = requiredString(*entry, "body_id", error);
      auto mapperShape = requiredString(*entry, "mapper_shape_id", error);
      auto status = requiredString(*entry, "support_status", error);
      if (!task || !bodyId || !mapperShape || !status)
        return false;
      BodyShapeCost cost;
      if (*status == "supported") {
        auto ii = entry->getNumber("predicted_ii");
        auto startup = entry->getNumber("startup_cycles");
        if (!ii || !startup || !std::isfinite(*ii) ||
            !std::isfinite(*startup) || *ii <= 0.0 || *startup <= 0.0) {
          error = "supported cost requires positive finite predicted_ii and "
                  "startup_cycles";
          return false;
        }
        cost = {*ii, *startup, true};
      } else if (*status != "unsupported") {
        error = "support_status must be supported or unsupported";
        return false;
      }
      CostKey key{task->str(), bodyId->str(), mapperShape->str()};
      if (!catalog_.emplace(std::move(key), cost).second) {
        error = "duplicate task/shape cost entry";
        return false;
      }
    }
    return true;
  }

  const BodyShapeCost *get(const TaskShapeChoice &choice, std::string &error) {
    // Across N candidates the same task-shape pair occurs many times.  The
    // first request is a miss copied from catalog_; all later requests return
    // the memoized value and count as hits.  Missing predictions are fatal:
    // silently inventing a fallback score would make candidate ranks unsound.
    CostKey key{choice.task, choice.bodyId, choice.shape.mapperShapeId};
    auto cached = cache_.find(key);
    if (cached != cache_.end()) {
      ++hits_;
      return &cached->second;
    }
    auto found = catalog_.find(key);
    if (found == catalog_.end()) {
      error = "missing cost for task=" + choice.task +
              ", body_id=" + choice.bodyId +
              ", mapper_shape=" + choice.shape.mapperShapeId;
      return nullptr;
    }
    ++misses_;
    auto inserted = cache_.emplace(key, found->second);
    return &inserted.first->second;
  }

  StringRef nameSpace() const { return namespace_; }
  StringRef catalogSha256() const { return catalogSha256_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t entries() const { return cache_.size(); }

private:
  std::string namespace_;
  std::string catalogSha256_;
  std::map<CostKey, BodyShapeCost> catalog_;
  std::map<CostKey, BodyShapeCost> cache_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

//===----------------------------------------------------------------------===//
// Pass 1: enumerate and freeze the complete shape space
//===----------------------------------------------------------------------===//

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
      llvm::cl::desc("Fail rather than publish a partial candidate manifest."),
      llvm::cl::init(1000000)};
  Option<int64_t> maxCgrasPerTask{
      *this, "max-cgras-per-task",
      llvm::cl::desc("Maximum rectangular physical footprint per task."),
      llvm::cl::init(4)};

  void runOnOperation() override {
    // Step 1: choose the Taskflow function and reject invalid safety limits.
    // This pass is observational with respect to IR; its only output is JSONL.
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

    // Step 2: snapshot the semantic task facts and enumerate the single-task
    // shape alphabet S from the physical architecture.
    FailureOr<SmallVector<TaskFact>> taskFacts = collectTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const neura::Architecture &architecture = neura::getArchitecture();
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

    // Step 3: the complete program space is S^T for T ordered tasks.  At this
    // shape-only layer, each task must fit the grid by itself; we intentionally
    // do not require the sum of all task footprints to fit simultaneously,
    // because later spatial/temporal scheduling may reuse the same CGRAs.
    // Check the exact size before opening the output.  Reaching max-candidates
    // is an error, never permission to keep a score-biased or prefix subset.
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
          // Step 4a: the header freezes every input needed to reconstruct S^T.
          // cost_queries is the de-duplicated domain the ML producer must
          // answer; fixed_axes makes the intentionally unimplemented search
          // dimensions explicit rather than leaving their meaning implicit.
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

          // Step 4b: depth-first recursion emits the Cartesian product in
          // task-major mixed-radix order.  No analytical value is available in
          // this pass, so every hard-valid combination is emitted.
          llvm::SHA256 idsDigest;
          uint64_t emitted = 0;
          SmallVector<TaskShapeChoice> selected;
          std::function<void(size_t)> visit = [&](size_t taskIndex) {
            if (taskIndex == taskFacts->size()) {
              Candidate candidate;
              candidate.choices = selected;
              candidate.id = candidateId(function, *architectureFingerprint,
                                         candidate.choices);
              writeJsonLine(os, candidateJson(candidate));
              updateIdDigest(idsDigest, candidate.id);
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
          // Step 4c: close the stream with both count and ordered-ID digest.
          // The common reader later recomputes these before scoring or replay.
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

//===----------------------------------------------------------------------===//
// Pass 2: score every frozen candidate, then form the shortlist
//===----------------------------------------------------------------------===//

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
      llvm::cl::desc("Shortlist size; zero selects every valid candidate."),
      llvm::cl::init(1)};

  void runOnOperation() override {
    // Step 1: establish that IR, candidate input, ML catalogue, and score
    // output are distinct and refer to the same function/architecture.
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
    FailureOr<std::string> architectureFingerprint =
        currentArchitectureFingerprint(error);
    if (failed(architectureFingerprint)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    BodyShapeCostCache costs;
    if (!costs.load(costFile.getValue(), func.getSymName(),
                    *architectureFingerprint, error)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // ranked contains only supported candidates, but it is deliberately not
    // sorted or truncated while readCandidateManifest is still streaming.
    // Every candidate, including unsupported ones, first gets a score record.
    SmallVector<RankedCandidate> ranked;
    ManifestHeader manifestHeader;
    ManifestFooter manifestFooter;
    uint64_t scoredCount = 0;
    uint64_t validCount = 0;
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          // The score header binds this ranking to the exact candidate schema,
          // architecture, model namespace, and bytes of the ML catalogue.
          llvm::json::Object scoreHeader;
          scoreHeader["record_type"] = "header";
          scoreHeader["schema_version"] = kScoreSchema.str();
          scoreHeader["candidate_schema_version"] = kCandidateSchema.str();
          scoreHeader["function"] = func.getSymName().str();
          scoreHeader["architecture_fingerprint"] = *architectureFingerprint;
          scoreHeader["cost_namespace"] = costs.nameSpace().str();
          scoreHeader["cost_catalog_sha256"] = costs.catalogSha256().str();
          scoreHeader["score_model"] = kScoreModel.str();
          writeJsonLine(os, std::move(scoreHeader));

          auto consume = [&](uint64_t, const Candidate &candidate,
                             std::string &consumeError) {
            // Shape-only score for one program candidate:
            //
            //   duration(task) = startup + II * (trip_count - 1)
            //   score(candidate) = max duration(task)
            //
            // This is only a compute bottleneck.  It must not be described as
            // the final taskflow interval until communication and temporal
            // scheduling are added in a later search-scope version.
            bool valid = true;
            double bottleneck = 0.0;
            std::string rejectReason;
            llvm::json::Array taskCosts;
            for (const TaskShapeChoice &choice : candidate.choices) {
              const BodyShapeCost *cost = costs.get(choice, consumeError);
              if (!cost)
                return false;
              llvm::json::Object taskCost;
              taskCost["task"] = choice.task;
              taskCost["mapper_shape_id"] = choice.shape.mapperShapeId;
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
                      ", mapper_shape=" + choice.shape.mapperShapeId;
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
            score["schema_version"] = kScoreSchema.str();
            score["candidate_id"] = candidate.id;
            score["valid"] = valid;
            score["task_costs"] = std::move(taskCosts);
            if (valid) {
              score["predicted_compute_bottleneck"] = bottleneck;
              ranked.push_back({candidate.id, bottleneck});
              ++validCount;
            } else {
              score["reject_reason"] = rejectReason;
            }
            writeJsonLine(os, std::move(score));
            ++scoredCount;
            return true;
          };

          // readCandidateManifest invokes consume exactly once per canonical
          // record and rejects an incomplete/reordered/tampered space.
          if (!readCandidateManifest(
                  candidateFile.getValue(), *taskFacts, func.getSymName(),
                  neura::getArchitecture(), *architectureFingerprint, consume,
                  manifestHeader, manifestFooter, error))
            return false;
          if (scoredCount != manifestFooter.candidateCount) {
            error = "not every frozen candidate was scored";
            return false;
          }

          // Only now, after scoredCount equals the proven manifest count, is
          // ranking allowed.  Candidate ID is a deterministic tie breaker, so
          // identical scores produce a reproducible top-k across processes.
          llvm::sort(ranked, [](const RankedCandidate &lhs,
                                const RankedCandidate &rhs) {
            if (lhs.score != rhs.score)
              return lhs.score < rhs.score;
            return lhs.id < rhs.id;
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

          // The footer is the external driver's control record.  It contains
          // shortlist IDs to materialize and enough provenance/counts to audit
          // that selection happened after full scoring.  The driver should run
          // the unchanged real pipeline once for each shortlist entry.
          llvm::json::Object cacheStats;
          cacheStats["hits"] = static_cast<int64_t>(costs.hits());
          cacheStats["misses"] = static_cast<int64_t>(costs.misses());
          cacheStats["entries"] = static_cast<int64_t>(costs.entries());
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema_version"] = kScoreSchema.str();
          footer["candidate_count"] =
              static_cast<int64_t>(manifestFooter.candidateCount);
          footer["scored_count"] = static_cast<int64_t>(scoredCount);
          footer["valid_count"] = static_cast<int64_t>(validCount);
          footer["candidate_ids_sha256"] = manifestFooter.candidateIdsSha256;
          footer["cost_catalog_sha256"] = costs.catalogSha256().str();
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

//===----------------------------------------------------------------------===//
// Pass 3: replay one shortlisted decision onto Taskflow IR
//===----------------------------------------------------------------------===//

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
      llvm::cl::desc("Candidate manifest index for testing/debugging."),
      llvm::cl::init(-1)};

  void runOnOperation() override {
    // Production callers should select by stable candidate-id.  Index exists
    // for debugging/replay tools, but requiring exactly one selector prevents
    // disagreement between two independently supplied choices.
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

    FailureOr<SmallVector<TaskFact>> taskFacts = collectTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    FailureOr<std::string> architectureFingerprint =
        currentArchitectureFingerprint(error);
    if (failed(architectureFingerprint)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Do not stop reading when the requested candidate is found.  The tail of
    // the file contains the footer/digest, and another matching record would
    // make selection ambiguous.  We remember the match while the common reader
    // validates the complete stream.
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
    if (!readCandidateManifest(candidateFile.getValue(), *taskFacts,
                               func.getSymName(), neura::getArchitecture(),
                               *architectureFingerprint, consume, header,
                               footer, error)) {
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

    // Mutation is intentionally small and delayed until the whole manifest,
    // checksum, current IR, architecture, and selected record are validated.
    // cgra_count/cgra_shape configure the existing downstream heuristic mapper;
    // no placement, route, II, or mapper result is fabricated here.
    OpBuilder builder(func.getContext());
    for (auto [task, choice] : llvm::zip(*taskFacts, selected->choices)) {
      task.op->setAttr("cgra_count",
                       builder.getI32IntegerAttr(choice.shape.cgraCount()));
      task.op->setAttr("cgra_shape",
                       builder.getStringAttr(choice.shape.irAttr()));
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

std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass() {
  return std::make_unique<EnumerateAnalyticalTaskCandidatesPass>();
}

std::unique_ptr<Pass> createScoreAnalyticalTaskCandidatesPass() {
  return std::make_unique<ScoreAnalyticalTaskCandidatesPass>();
}

std::unique_ptr<Pass> createMaterializeAnalyticalTaskCandidatePass() {
  return std::make_unique<MaterializeAnalyticalTaskCandidatePass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
