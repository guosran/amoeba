//===- AnalyticalTaskDSESupport.cpp - Shared task DSE support ------------===//
//
// Implements the records and file protocol shared by the static analytical
// task-DSE passes.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  return std::to_string(rows) + "x" + std::to_string(cols);
}

// Resolves the trip count that analytical DSE stores with each task. An
// explicit positive `trip_count` is authoritative; otherwise, a static
// Taskflow counter chain supplies the count. A task with no Taskflow counter
// represents one execution at this layer, while a dynamic or invalid counter
// fails because treating it as one would produce an incorrect score.
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

  FailureOr<std::optional<int64_t>> inferred =
      inferStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit positive trip_count or resolve the counter "
             "bounds first";
    return failure();
  }
  // A task without a Taskflow counter executes once in the task-level model.
  return inferred->value_or(1);
}

// Collects task names, operations, and static trip counts in walk order. The
// order is the task axis used by the candidate Cartesian product, so duplicate
// names are rejected before they can make a cost lookup ambiguous.
FailureOr<SmallVector<TaskFact>> collectTaskFacts(func::FuncOp func,
                                                  std::string &error) {
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
    tasks.push_back({task, std::move(name), *tripCount});
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

// Formats a mapper tile rectangle as its stable, human-readable key. A
// physical 1x2 shape on a 4x4-tile CGRA therefore becomes `rect-4x8`.
static std::string toMapperShapeString(int64_t mapperRows, int64_t mapperCols) {
  return "rect-" + std::to_string(mapperRows) + "x" +
         std::to_string(mapperCols);
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The order is deterministic and
// is later used as the mixed-radix alphabet for candidate IDs and validation.
SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                 int64_t gridCols,
                                                 int64_t perCgraRows,
                                                 int64_t perCgraCols,
                                                 int64_t maxCgrasPerTask) {
  SmallVector<RectShape> result;
  if (gridRows <= 0 || gridCols <= 0 || perCgraRows <= 0 || perCgraCols <= 0 ||
      maxCgrasPerTask <= 0)
    return result;

  const int64_t gridSize =
      gridRows > std::numeric_limits<int64_t>::max() / gridCols
          ? std::numeric_limits<int64_t>::max()
          : gridRows * gridCols;
  const int64_t maxCount = std::min(maxCgrasPerTask, gridSize);
  for (int64_t count = 1; count <= maxCount; ++count) {
    for (const CgraShape &physicalShape : taskflow::getRectangularShapes(
             static_cast<int>(count), static_cast<int>(gridRows),
             static_cast<int>(gridCols))) {
      if (physicalShape.rows >
              std::numeric_limits<int64_t>::max() / perCgraRows ||
          physicalShape.cols >
              std::numeric_limits<int64_t>::max() / perCgraCols)
        continue;
      int64_t mapperRows = physicalShape.rows * perCgraRows;
      int64_t mapperCols = physicalShape.cols * perCgraCols;
      result.push_back(
          {physicalShape.rows, physicalShape.cols, mapperRows, mapperCols});
    }
  }
  return result;
}

// Formats the deterministic candidate ID assigned by enumeration. For
// example, the first candidate is `candidate-0`, and the next is
// `candidate-1`; the numeric suffix is the canonical mixed-radix index.
std::string makeSequentialCandidateId(uint64_t index) {
  return "candidate-" + std::to_string(index);
}

// Serializes the shape fields used by candidate records. The explicit tile
// dimensions are the only mapper-shape truth; a `rect-4x8` string is produced
// only for diagnostics when needed.
static llvm::json::Object shapeJson(const RectShape &shape) {
  llvm::json::Object object;
  object["kind"] = "rect";
  object["rows"] = shape.rows;
  object["cols"] = shape.cols;
  object["cgra_count"] = shape.cgraCount();
  object["cgra_shape"] = shape.toCgraShapeAttrValue();
  object["mapper_tile_rows"] = shape.mapperRows;
  object["mapper_tile_cols"] = shape.mapperCols;
  return object;
}

// Serializes one candidate while preserving task order. The candidate ID is a
// sequential index, so no task-body identity or file-derived metadata is
// required to interpret it within its validated v2 manifest.
llvm::json::Object candidateJson(const Candidate &candidate) {
  llvm::json::Array choices;
  for (const TaskShapeChoice &choice : candidate.choices) {
    llvm::json::Object record;
    record["task"] = choice.task;
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

// Writes one JSON object as a single JSONL record.
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

// Publishes a complete output atomically so a downstream pass never reads a
// partially enumerated or partially scored file.
bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error) {
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
    ok = writeBody(os);
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

// Normalizes a path for the output-collision check while tolerating paths that
// do not exist yet.
static SmallString<256> normalizedPath(StringRef path) {
  SmallString<256> result(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(result))
    return SmallString<256>(path);
  llvm::sys::path::remove_dots(result, /*remove_dot_dot=*/true);
  return result;
}

// Checks whether two paths identify the same file or normalized pathname.
bool samePath(StringRef lhs, StringRef rhs) {
  bool equivalent = false;
  if (!llvm::sys::fs::equivalent(lhs, rhs, equivalent) && equivalent)
    return true;
  return normalizedPath(lhs) == normalizedPath(rhs);
}

// Selects the requested Taskflow function, or infers it when exactly one
// function contains tasks. This keeps every file-producing pass consistent.
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module, StringRef requested,
                                           std::string &error) {
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

// Reads a required JSON string and reports a field-specific error.
static std::optional<StringRef> requiredString(const llvm::json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value)
    error = "missing or invalid string field \"" + key.str() + "\"";
  return value;
}

// Reads a required JSON integer and reports a field-specific error.
static std::optional<int64_t> requiredInteger(const llvm::json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

// Multiplies two positive counts while detecting overflow before the product.
static std::optional<int64_t> checkedPositiveProduct(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::nullopt;
  return lhs * rhs;
}

// Parses the v2 manifest header, including the explicit architecture
// dimensions that describe how physical CGRAs map to tiles. The dimensions
// are later compared directly with `getArchitecture()` getters.
static bool parseHeader(const llvm::json::Object &object,
                        ManifestHeader &header, std::string &error) {
  std::optional<StringRef> schema =
      requiredString(object, "schema_version", error);
  std::optional<StringRef> function = requiredString(object, "function", error);
  std::optional<StringRef> scope =
      requiredString(object, "search_scope", error);
  std::optional<StringRef> policy =
      requiredString(object, "shape_policy", error);
  const llvm::json::Object *architecture = object.getObject("architecture");
  if (!schema || !function || !scope || !policy || !architecture)
    return false;
  if (*schema != kCandidateSchema || *scope != kSearchScope ||
      *policy != kShapePolicy) {
    error = "unsupported candidate manifest contract";
    return false;
  }

  auto gridRows = requiredInteger(*architecture, "grid_rows", error);
  auto gridCols = requiredInteger(*architecture, "grid_cols", error);
  auto perRows = requiredInteger(*architecture, "per_cgra_tile_rows", error);
  auto perCols = requiredInteger(*architecture, "per_cgra_tile_cols", error);
  auto maxCgras = requiredInteger(object, "max_cgras_per_task", error);
  if (!gridRows || !gridCols || !perRows || !perCols || !maxCgras)
    return false;
  header = {function->str(), *gridRows, *gridCols,
            *perRows,        *perCols,  *maxCgras};
  if (header.gridRows <= 0 || header.gridCols <= 0 || header.perCgraRows <= 0 ||
      header.perCgraCols <= 0 || header.maxCgrasPerTask <= 0) {
    error = "candidate manifest dimensions must be positive";
    return false;
  }
  return true;
}

// Verifies that the manifest task list has the same names and static trip
// counts as the current IR.
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
    auto tripCount = requiredInteger(*record, "trip_count", error);
    if (!name || !tripCount)
      return false;
    if (*name != tasks[index].name || *tripCount != tasks[index].tripCount) {
      error = "candidate manifest header task facts do not match current IR";
      return false;
    }
  }
  return true;
}

// Parses one rectangular shape and checks its redundant fields against the
// dimensions. The explicit tile rows and columns are the mapper-shape truth.
static bool parseShape(const llvm::json::Object &object, RectShape &shape,
                       std::string &error) {
  auto kind = requiredString(object, "kind", error);
  auto rows = requiredInteger(object, "rows", error);
  auto cols = requiredInteger(object, "cols", error);
  auto count = requiredInteger(object, "cgra_count", error);
  auto irShape = requiredString(object, "cgra_shape", error);
  auto mapperRows = requiredInteger(object, "mapper_tile_rows", error);
  auto mapperCols = requiredInteger(object, "mapper_tile_cols", error);
  if (!kind || !rows || !cols || !count || !irShape || !mapperRows ||
      !mapperCols)
    return false;
  std::optional<int64_t> computedCount = checkedPositiveProduct(*rows, *cols);
  if (*kind != "rect" || !computedCount || *count != *computedCount ||
      *mapperRows <= 0 || *mapperCols <= 0) {
    error = "candidate contains a non-rectangular or invalid shape";
    return false;
  }
  shape = {*rows, *cols, *mapperRows, *mapperCols};
  if (*irShape != shape.toCgraShapeAttrValue()) {
    error = "candidate physical shape label does not match its dimensions";
    return false;
  }
  return true;
}

// Compares the physical and mapper dimensions of two rectangles.
static bool sameShape(const RectShape &lhs, const RectShape &rhs) {
  return std::tie(lhs.rows, lhs.cols, lhs.mapperRows, lhs.mapperCols) ==
         std::tie(rhs.rows, rhs.cols, rhs.mapperRows, rhs.mapperCols);
}

// Parses one candidate record and verifies task names and static trip counts.
// Candidate ordering and the sequential ID are checked by the stream reader,
// which knows the record's canonical mixed-radix index.
static bool parseCandidate(const llvm::json::Object &object,
                           ArrayRef<TaskFact> tasks, Candidate &candidate,
                           std::string &error) {
  auto schema = requiredString(object, "schema_version", error);
  auto id = requiredString(object, "candidate_id", error);
  const llvm::json::Array *records = object.getArray("task_shapes");
  if (!schema || !id || !records)
    return false;
  if (*schema != kCandidateSchema || records->size() != tasks.size()) {
    error = "candidate schema or task count does not match its manifest";
    return false;
  }

  candidate.id = id->str();
  candidate.choices.clear();
  for (auto [index, value] : llvm::enumerate(*records)) {
    const llvm::json::Object *record = value.getAsObject();
    if (!record) {
      error = "task_shapes entry is not an object";
      return false;
    }
    auto taskName = requiredString(*record, "task", error);
    auto tripCount = requiredInteger(*record, "trip_count", error);
    const llvm::json::Object *shapeObject = record->getObject("shape");
    if (!taskName || !tripCount || !shapeObject)
      return false;
    const TaskFact &task = tasks[index];
    if (*taskName != task.name || *tripCount != task.tripCount) {
      error = "candidate task facts do not match the current IR";
      return false;
    }
    RectShape shape;
    if (!parseShape(*shapeObject, shape, error))
      return false;
    candidate.choices.push_back({task.name, task.tripCount, std::move(shape)});
  }
  return true;
}

// Computes the expected size of the task-shape Cartesian product.
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

// Verifies the canonical mixed-radix shape assignment at one candidate index.
// The last task changes fastest, so index 1 for two tasks means
// `(task0=shape0, task1=shape1)`.
static bool validateCandidateAtIndex(uint64_t index, ArrayRef<TaskFact> tasks,
                                     ArrayRef<RectShape> shapes,
                                     const Candidate &candidate,
                                     std::string &error) {
  if (candidate.id != makeSequentialCandidateId(index)) {
    error = "candidate ID does not match its canonical manifest index";
    return false;
  }
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

// Compares manifest architecture dimensions directly with the current
// architecture object. Reading architecture.yaml uses these getters; no file
// fingerprint or other file-derived metadata is needed for this protocol.
static bool architectureMatches(const ManifestHeader &header,
                                const ::mlir::neura::Architecture &architecture,
                                std::string &error) {
  if (header.gridRows != architecture.getMultiCgraRows() ||
      header.gridCols != architecture.getMultiCgraColumns() ||
      header.perCgraRows != architecture.getPerCgraRows() ||
      header.perCgraCols != architecture.getPerCgraColumns()) {
    error = "candidate manifest architecture dimensions do not match current "
            "Neura architecture";
    return false;
  }
  return true;
}

// Streams and validates a complete candidate manifest before forwarding each
// candidate to the scoring or materialization callback. The footer verifies
// both the emitted count and the mathematically expected Cartesian-product
// size; every candidate verifies its sequential ID and mixed-radix shape.
bool readCandidateManifest(StringRef path, ArrayRef<TaskFact> tasks,
                           StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error) {
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
      if (!architectureMatches(header, architecture, error))
        return false;
      legalShapes = enumerateStaticRectShapes(
          header.gridRows, header.gridCols, header.perCgraRows,
          header.perCgraCols, header.maxCgrasPerTask);
      FailureOr<uint64_t> computed =
          expectedCandidateCount(legalShapes.size(), tasks.size(), error);
      if (failed(computed))
        return false;
      declaredSpaceCount = *computed;
      sawHeader = true;
      continue;
    }
    if (*recordType == "candidate") {
      if (!sawHeader || sawFooter) {
        error = "candidate record is outside header/footer";
        return false;
      }
      Candidate candidate;
      if (!parseCandidate(*object, tasks, candidate, error) ||
          !validateCandidateAtIndex(count, tasks, legalShapes, candidate,
                                    error) ||
          !consume(count, candidate, error))
        return false;
      ++count;
      continue;
    }
    if (*recordType == "footer") {
      auto schema = requiredString(*object, "schema_version", error);
      auto expectedCount = requiredInteger(*object, "candidate_count", error);
      if (!sawHeader || sawFooter || !schema || !expectedCount ||
          *schema != kCandidateSchema || *expectedCount < 0) {
        if (error.empty())
          error = "invalid candidate manifest footer";
        return false;
      }
      if (static_cast<uint64_t>(*expectedCount) != count ||
          count != declaredSpaceCount) {
        error = "candidate manifest count does not match its declared shape "
                "space";
        return false;
      }
      footer = {count};
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

// Loads and validates the v2 task-shape cost catalogue. Each entry is keyed by
// task name and explicit mapper tile dimensions, so a `rect-4x8` label remains
// readable without becoming a correctness dependency.
bool TaskShapeCostCache::load(StringRef path, StringRef expectedFunction,
                              std::string &error) {
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
  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "cost catalogue must be a JSON object";
    return false;
  }
  auto schema = requiredString(*root, "schema_version", error);
  auto function = requiredString(*root, "function", error);
  auto modelNamespace = requiredString(*root, "namespace", error);
  llvm::json::Array *entries = root->getArray("entries");
  if (!schema || !function || !modelNamespace || !entries)
    return false;
  if (*schema != kCostSchema || *function != expectedFunction ||
      modelNamespace->empty()) {
    error = "cost catalogue schema/function/namespace mismatch";
    return false;
  }

  namespace_ = modelNamespace->str();
  catalog_.clear();
  cache_.clear();
  hits_ = 0;
  misses_ = 0;
  for (llvm::json::Value &value : *entries) {
    llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "cost entry is not an object";
      return false;
    }
    auto task = requiredString(*entry, "task", error);
    auto mapperRows = requiredInteger(*entry, "mapper_tile_rows", error);
    auto mapperCols = requiredInteger(*entry, "mapper_tile_cols", error);
    auto status = requiredString(*entry, "support_status", error);
    if (!task || !mapperRows || !mapperCols || !status)
      return false;
    if (*mapperRows <= 0 || *mapperCols <= 0) {
      error = "cost entry mapper tile dimensions must be positive";
      return false;
    }
    TaskShapeCost cost;
    if (*status == "supported") {
      auto ii = entry->getNumber("predicted_ii");
      auto startup = entry->getNumber("startup_cycles");
      if (!ii || !startup || !std::isfinite(*ii) || !std::isfinite(*startup) ||
          *ii <= 0.0 || *startup <= 0.0) {
        error = "supported cost requires positive finite predicted_ii and "
                "startup_cycles";
        return false;
      }
      cost = {*ii, *startup, true};
    } else if (*status != "unsupported") {
      error = "support_status must be supported or unsupported";
      return false;
    }
    CostKey key{task->str(), *mapperRows, *mapperCols};
    if (!catalog_.emplace(std::move(key), cost).second) {
      error = "duplicate task/mapper-shape cost entry";
      return false;
    }
  }
  return true;
}

// Looks up one task and mapper rectangle, recording hits and misses for the
// scorer's audit footer. Repeated candidates reuse the same cached value.
const TaskShapeCost *TaskShapeCostCache::get(const TaskShapeChoice &choice,
                                             std::string &error) {
  CostKey key{choice.task, choice.shape.mapperRows, choice.shape.mapperCols};
  auto cached = cache_.find(key);
  if (cached != cache_.end()) {
    ++hits_;
    return &cached->second;
  }
  auto found = catalog_.find(key);
  if (found == catalog_.end()) {
    error =
        "missing cost for task=" + choice.task + ", mapper_shape=" +
        toMapperShapeString(choice.shape.mapperRows, choice.shape.mapperCols);
    return nullptr;
  }
  ++misses_;
  auto inserted = cache_.emplace(key, found->second);
  return &inserted.first->second;
}

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir
