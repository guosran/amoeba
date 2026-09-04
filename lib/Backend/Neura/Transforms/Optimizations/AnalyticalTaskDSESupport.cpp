//===- AnalyticalTaskDSESupport.cpp - Shared task DSE support ------------===//
//
// Implements the internal records and file protocol shared by the analytical
// task-DSE passes.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

std::string RectShape::irAttr() const {
  return std::to_string(rows) + "x" + std::to_string(cols);
}

std::string sha256Hex(StringRef text) {
  llvm::SHA256 hasher;
  hasher.update(text);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

void updateCandidateIdDigest(llvm::SHA256 &hasher, StringRef candidateId) {
  hasher.update(candidateId);
  hasher.update("\n");
}

// Binds every manifest and task-shape cost catalogue to the exact architecture
// specification selected by the Amoeba driver. Hashing the file intentionally
// invalidates stale scores after even semantically equivalent edits.
FailureOr<std::string> currentArchitectureFingerprint(std::string &error) {
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
  return "sha256:" + sha256Hex((*buffer)->getBuffer());
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
         sha256Hex(canonical).substr(0, 10);
}

static std::string taskBodyId(TaskflowTaskOp task) {
  if (auto explicitId = task->getAttrOfType<StringAttr>("analytical_body_id"))
    if (!explicitId.getValue().empty())
      return explicitId.getValue().str();

  // Hashes the computation and interface without transient decisions or
  // measurements. This preserves reuse of the same task-shape query after
  // materialization or profiling.
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
  return "sha256:" + sha256Hex(printed);
}

// Resolves an explicit trip count before consulting shared counter analysis.
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
      computeTaskflowCounterTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit trip_count";
    return failure();
  }
  return inferred->value_or(1);
}

FailureOr<SmallVector<TaskFact>> collectTaskFacts(func::FuncOp func,
                                                  std::string &error) {
  // Collects the task axis of the Cartesian product in deterministic order.
  // Rejects duplicate names before they make cost keys or replay ambiguous.
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

// Extends the shared physical-shape enumeration with mapper dimensions.
SmallVector<RectShape> enumerateRectShapes(int64_t gridRows, int64_t gridCols,
                                           int64_t perCgraRows,
                                           int64_t perCgraCols,
                                           int64_t maxCgrasPerTask) {
  SmallVector<RectShape> result;
  const int64_t maxCount = std::min(maxCgrasPerTask, gridRows * gridCols);
  for (int64_t count = 1; count <= maxCount; ++count) {
    for (const CgraShape &physicalShape : taskflow::getRectangularShapes(
             count, static_cast<int>(gridRows), static_cast<int>(gridCols))) {
      int64_t mapperRows = physicalShape.rows * perCgraRows;
      int64_t mapperCols = physicalShape.cols * perCgraCols;
      result.push_back({physicalShape.rows, physicalShape.cols, mapperRows,
                        mapperCols, mapperShapeId(mapperRows, mapperCols)});
    }
  }
  return result;
}

static void appendIdentityField(std::string &out, StringRef key,
                                StringRef value) {
  // Length-prefixes every value so concatenation cannot alias, for example,
  // ("ab", "c") and ("a", "bc") produce different byte streams.
  out += key.str();
  out += ":";
  out += std::to_string(value.size());
  out += ":";
  out += value.str();
  out += "\n";
}

std::string makeCandidateId(StringRef function,
                            StringRef architectureFingerprint,
                            ArrayRef<TaskShapeChoice> choices) {
  // Hashes an explicit canonical stream because JSON field order and
  // whitespace are interchange details. This defines which semantic changes
  // invalidate a candidate ID.
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
  return sha256Hex(identity);
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

llvm::json::Object candidateJson(const Candidate &candidate) {
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

void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error) {
  // Prevents a downstream driver from mistaking a partial file for a pruned
  // search space. Writes a sibling temporary and renames it only after success.
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

static SmallString<256> normalizedPath(StringRef path) {
  SmallString<256> result(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(result))
    return SmallString<256>(path);
  llvm::sys::path::remove_dots(result, /*remove_dot_dot=*/true);
  return result;
}

bool samePath(StringRef lhs, StringRef rhs) {
  bool equivalent = false;
  if (!llvm::sys::fs::equivalent(lhs, rhs, equivalent) && equivalent)
    return true;
  return normalizedPath(lhs) == normalizedPath(rhs);
}

FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module, StringRef requested,
                                           std::string &error) {
  // Requires an unambiguous Taskflow function for module-level passes. Allows
  // explicit selection while keeping the single-function case convenient.
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

// Enforces stricter validation than ordinary configuration-file parsing. The
// manifest separates exhaustive enumeration from scoring and materialization,
// so accepting altered records would silently introduce heuristic pruning.
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
  // Gives later extensions an explicit migration point through versioned
  // schemas instead of reinterpreting an old manifest.
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
  // Proceeds from cheap structural validation to task facts, dimension
  // arithmetic, shape-family membership, and stable-ID recomputation.
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
  candidate.id = makeCandidateId(
      header.function, header.architectureFingerprint, candidate.choices);
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

// Treats enumeration order as part of the frozen contract. Checking the exact
// record at each index detects omissions, duplicates, and reorderings even when
// an editor recomputes the footer digest.
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
                                const ::mlir::neura::Architecture &architecture,
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

bool readCandidateManifest(StringRef path, ArrayRef<TaskFact> tasks,
                           StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           StringRef architectureFingerprint,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error) {
  // Streams candidates instead of retaining the Cartesian product in memory.
  // Gives the common reader ownership of completeness and provenance checks.
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
      // Requires the header first and reconstructs both the expected product
      // size and canonical mixed-radix ordering from it.
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
      // Prevents an editor from removing or reordering records and merely
      // recomputing the digest by validating each candidate at its index.
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
      updateCandidateIdDigest(idsDigest, candidate.id);
      ++count;
      continue;
    }
    if (*recordType == "footer") {
      // Closes the stream only when its count, expected product size, and
      // ordered-ID digest agree.
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

bool BodyShapeCostCache::load(StringRef path, StringRef expectedFunction,
                              StringRef expectedArchitectureFingerprint,
                              std::string &error) {
  // Validates the entire catalogue eagerly, including unused entries, because
  // duplicates and malformed predictions indicate a producer error.
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
  catalogSha256_ = sha256Hex((*buffer)->getBuffer());
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
    CostKey key{task->str(), bodyId->str(), mapperShape->str()};
    if (!catalog_.emplace(std::move(key), cost).second) {
      error = "duplicate task/shape cost entry";
      return false;
    }
  }
  return true;
}

const BodyShapeCost *BodyShapeCostCache::get(const TaskShapeChoice &choice,
                                             std::string &error) {
  // Counts the first request as a miss copied from the validated catalogue and
  // serves every repeated task-shape request from the per-pass cache.
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

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir
