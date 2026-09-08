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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <functional>
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

// Returns the lowercase SHA-256 used by the Python adapter. Hashes are taken
// over raw file bytes so changes in comments or formatting conservatively
// invalidate previously generated artifacts.
static std::string sha256(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static bool isSha256(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// Fingerprints the exact YAML selected by --architecture-spec. Dimensions
// alone are insufficient because two same-sized machines can have different
// FU, memory, latency, or routing capabilities.
FailureOr<std::string> currentArchitectureSha256(std::string &error) {
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
  return sha256((*buffer)->getBuffer());
}

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  return taskflow::formatRectangularCgraShape(rows, cols);
}

// Resolves the trip count that analytical DSE stores with each task. An
// explicit positive `trip_count` is authoritative; otherwise, a static
// Taskflow counter chain supplies the count. A task with no Taskflow counter
// represents one execution at this layer, while a dynamic or invalid counter
// fails because treating it as one would produce an incorrect score.
static FailureOr<int64_t> resolveAnalyticalTripCount(TaskflowTaskOp task,
                                                     std::string &error) {
  // TODO: add an explicit runtime-parameter contract before admitting dynamic
  // trip counts. The current static-shape protocol must fail rather than guess.
  FailureOr<std::optional<int64_t>> inferred =
      resolveStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit positive trip_count or resolve the counter "
             "bounds first";
    return failure();
  }
  // A task without a Taskflow counter executes once in the task-level model.
  return inferred->value_or(1);
}

// Produces a stable identity for the current task computation. We deliberately
// remove DSE outputs and measurements so materializing a shape does not make
// an otherwise identical task look new. A body edit, however, changes this
// hash and invalidates the old candidate manifest before scoring.
static std::string taskBodySha256(TaskflowTaskOp task) {
  Operation *clone = task->clone();
  auto destroyClone = llvm::make_scope_exit([&] { clone->destroy(); });
  clone->setAttr("task_name",
                 StringAttr::get(task.getContext(), "__analytical_task__"));
  for (StringRef attribute :
       {"trip_count", "cgra_count", "cgra_shape", "compiled_ii", "profile_info",
        "task_orchestration_info", "replicas", "tiling", "est_latency"})
    clone->removeAttr(attribute);
  clone->removeAttr("amoeba.analytical_shape_orientation_fixed");
  // This attribute is an exported copy of the hash being computed. Excluding
  // it keeps the identity stable when an already-bound IR is enumerated again.
  clone->removeAttr(kSourceTaskBodyShaAttr);
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  OpPrintingFlags flags;
  flags.printGenericOpForm().useLocalScope();
  clone->print(stream, flags);
  stream.flush();
  return sha256(printed);
}

// Collects task names, operations, and static trip counts in walk order. The
// order is the task axis used by shape-tuple enumeration, so duplicate names
// are rejected before they can make a cost lookup ambiguous.
FailureOr<SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error) {
  SmallVector<TaskFact> tasks;
  llvm::StringSet<> names;
  for (TaskflowTaskOp task : collectTaskflowTasks(func)) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return failure();
    }
    FailureOr<int64_t> tripCount = resolveAnalyticalTripCount(task, error);
    if (failed(tripCount))
      return failure();
    tasks.push_back({task, std::move(name), taskBodySha256(task), *tripCount});
  }
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
  // TODO: If allocation dimensions ever become runtime parameters, represent
  // them symbolically and define how a finite DSE domain is bounded. The
  // current manifest intentionally contains only concrete integer rectangles;
  // for example, `1x2` is legal while `1xN` is not a candidate shape.
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

// Tries to place the fixed-orientation rectangles exactly. This is a small
// backtracking search over physical CGRA cells, not a solver and not the
// downstream placement heuristic. Sorting large rectangles first only changes
// search speed; it does not remove any legal placement.
static bool placeRectangles(size_t rectangleIndex,
                            ArrayRef<RectShape> rectangles, int64_t gridRows,
                            int64_t gridCols,
                            MutableArrayRef<uint8_t> occupied) {
  if (rectangleIndex == rectangles.size())
    return true;

  const RectShape &shape = rectangles[rectangleIndex];
  for (int64_t originRow = 0; originRow + shape.rows <= gridRows; ++originRow) {
    for (int64_t originCol = 0; originCol + shape.cols <= gridCols;
         ++originCol) {
      bool overlaps = false;
      for (int64_t row = 0; row < shape.rows && !overlaps; ++row) {
        for (int64_t col = 0; col < shape.cols; ++col) {
          size_t cell = static_cast<size_t>((originRow + row) * gridCols +
                                            originCol + col);
          if (occupied[cell]) {
            overlaps = true;
            break;
          }
        }
      }
      if (overlaps)
        continue;

      for (int64_t row = 0; row < shape.rows; ++row)
        for (int64_t col = 0; col < shape.cols; ++col)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 1;
      if (placeRectangles(rectangleIndex + 1, rectangles, gridRows, gridCols,
                          occupied))
        return true;
      for (int64_t row = 0; row < shape.rows; ++row)
        for (int64_t col = 0; col < shape.cols; ++col)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 0;
    }
  }
  return false;
}

// TODO: Replace this temporary simultaneous-packing filter with analytical
// spatial-temporal scheduling. The future scheduler should retain every tuple
// whose individual shapes fit the grid, then evaluate placement, temporal
// reuse, and communication jointly. For example, two 4x4 tasks do not fit on a
// 4x4 grid at the same time, but are legal when the second task reuses the grid
// after the first task finishes. Until that scheduler exists, this function
// conservatively requires every task rectangle to be resident simultaneously.
//
// The area check below is only a cheap necessary condition. The backtracking
// placement is still required because, for example, a horizontal 1x4 rectangle
// and a vertical 4x1 rectangle have total area eight but cannot coexist on a
// 4x4 grid: they must intersect in one cell.
static bool canPackSimultaneously(ArrayRef<RectShape> selected,
                                  int64_t gridRows, int64_t gridCols) {
  if (gridRows <= 0 || gridCols <= 0 ||
      gridRows > std::numeric_limits<int64_t>::max() / gridCols)
    return false;
  const int64_t gridArea = gridRows * gridCols;
  int64_t selectedArea = 0;
  SmallVector<RectShape> largestFirst(selected.begin(), selected.end());
  for (const RectShape &shape : largestFirst) {
    if (shape.rows <= 0 || shape.cols <= 0 || shape.rows > gridRows ||
        shape.cols > gridCols || shape.rows > gridArea / shape.cols ||
        selectedArea > gridArea - shape.cgraCount())
      return false;
    selectedArea += shape.cgraCount();
  }
  llvm::sort(largestFirst, [](const RectShape &lhs, const RectShape &rhs) {
    if (lhs.cgraCount() != rhs.cgraCount())
      return lhs.cgraCount() > rhs.cgraCount();
    if (std::max(lhs.rows, lhs.cols) != std::max(rhs.rows, rhs.cols))
      return std::max(lhs.rows, lhs.cols) > std::max(rhs.rows, rhs.cols);
    return std::tie(lhs.rows, lhs.cols) > std::tie(rhs.rows, rhs.cols);
  });

  SmallVector<uint8_t> occupied(static_cast<size_t>(gridArea), 0);
  return placeRectangles(0, largestFirst, gridRows, gridCols, occupied);
}

bool ConcurrentPackingCache::canPack(ArrayRef<RectShape> shapes) {
  Key key;
  key.reserve(shapes.size());
  for (const RectShape &shape : shapes)
    key.push_back({shape.rows, shape.cols});
  llvm::sort(key, [](const auto &lhs, const auto &rhs) {
    const int64_t lhsArea = lhs.first * lhs.second;
    const int64_t rhsArea = rhs.first * rhs.second;
    if (lhsArea != rhsArea)
      return lhsArea > rhsArea;
    if (std::max(lhs.first, lhs.second) != std::max(rhs.first, rhs.second))
      return std::max(lhs.first, lhs.second) > std::max(rhs.first, rhs.second);
    return lhs > rhs;
  });

  auto found = results_.find(key);
  if (found != results_.end())
    return found->second;
  bool result = canPackSimultaneously(shapes, gridRows_, gridCols_);
  results_.emplace(std::move(key), result);
  return result;
}

bool visitConcurrentlyPackableShapeTuples(size_t taskCount,
                                          ArrayRef<RectShape> shapes,
                                          ConcurrentPackingCache &packing,
                                          ShapeIndexTupleConsumer consume) {
  const int64_t gridRows = packing.gridRows();
  const int64_t gridCols = packing.gridCols();
  if (taskCount == 0 || shapes.empty() || gridRows <= 0 || gridCols <= 0 ||
      gridRows > std::numeric_limits<int64_t>::max() / gridCols)
    return true;
  const int64_t gridArea = gridRows * gridCols;
  // Every supported task consumes at least one physical CGRA. This prevents a
  // large task list from expanding an obviously empty Cartesian product.
  if (taskCount > static_cast<size_t>(gridArea))
    return true;

  SmallVector<size_t> selectedIndices;
  SmallVector<RectShape> selectedShapes;
  uint64_t validIndex = 0;
  std::function<bool(size_t, int64_t)> visit = [&](size_t taskIndex,
                                                   int64_t selectedArea) {
    if (taskIndex == taskCount) {
      if (!packing.canPack(selectedShapes))
        return true;
      if (!consume(validIndex, selectedIndices))
        return false;
      ++validIndex;
      return true;
    }

    // Shapes remain in their declared deterministic order. The area test
    // removes only tuples that cannot possibly fit; geometry is checked
    // exactly after one shape has been chosen for every task.
    for (auto [shapeIndex, shape] : llvm::enumerate(shapes)) {
      const int64_t area = shape.cgraCount();
      if (area <= 0 || area > gridArea - selectedArea)
        continue;
      selectedIndices.push_back(shapeIndex);
      selectedShapes.push_back(shape);
      if (!visit(taskIndex + 1, selectedArea + area))
        return false;
      selectedShapes.pop_back();
      selectedIndices.pop_back();
    }
    return true;
  };
  return visit(0, 0);
}

// Formats the deterministic candidate ID assigned by enumeration. For
// example, the first candidate is `candidate-0`, and the next is
// `candidate-1`. The numeric suffix indexes only concurrently packable tuples,
// so IDs stay contiguous after impossible shape assignments are removed.
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
// required to interpret it within its validated manifest.
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
  record["schema"] = kCandidateSchema.str();
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

// Parses the manifest header, including the explicit architecture
// dimensions that describe how physical CGRAs map to tiles. The dimensions
// are later compared directly with `getArchitecture()` getters.
static bool parseHeader(const llvm::json::Object &object,
                        ManifestHeader &header, std::string &error) {
  std::optional<StringRef> schema = requiredString(object, "schema", error);
  std::optional<StringRef> function = requiredString(object, "function", error);
  std::optional<StringRef> scope =
      requiredString(object, "search_scope", error);
  std::optional<StringRef> policy =
      requiredString(object, "shape_policy", error);
  std::optional<StringRef> capacityPolicy =
      requiredString(object, "spatial_capacity_policy", error);
  const llvm::json::Object *architecture = object.getObject("architecture");
  if (!schema || !function || !scope || !policy || !capacityPolicy ||
      !architecture)
    return false;
  if (*schema != kCandidateSchema || *scope != kSearchScope ||
      *policy != kShapePolicy || *capacityPolicy != kSpatialCapacityPolicy) {
    error = "unsupported candidate manifest contract";
    return false;
  }

  auto gridRows = requiredInteger(*architecture, "grid_rows", error);
  auto gridCols = requiredInteger(*architecture, "grid_cols", error);
  auto perRows = requiredInteger(*architecture, "per_cgra_tile_rows", error);
  auto perCols = requiredInteger(*architecture, "per_cgra_tile_cols", error);
  auto architectureSha = requiredString(*architecture, "spec_sha256", error);
  auto maxCgras = requiredInteger(object, "max_cgras_per_task", error);
  if (!gridRows || !gridCols || !perRows || !perCols || !architectureSha ||
      !maxCgras)
    return false;
  header = {function->str(), architectureSha->str(),
            *gridRows,       *gridCols,
            *perRows,        *perCols,
            *maxCgras};
  if (header.gridRows <= 0 || header.gridCols <= 0 || header.perCgraRows <= 0 ||
      header.perCgraCols <= 0 || header.maxCgrasPerTask <= 0) {
    error = "candidate manifest dimensions must be positive";
    return false;
  }
  if (!isSha256(header.architectureSha256)) {
    error = "candidate manifest architecture SHA-256 is invalid";
    return false;
  }
  return true;
}

// Verifies that the manifest task list has the same names, canonical bodies,
// and static trip counts as the current IR.
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
    auto bodySha = requiredString(*record, "body_sha256", error);
    auto tripCount = requiredInteger(*record, "trip_count", error);
    if (!name || !bodySha || !tripCount)
      return false;
    if (!isSha256(*bodySha) || *name != tasks[index].name ||
        *bodySha != tasks[index].bodySha256 ||
        *tripCount != tasks[index].tripCount) {
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
  // TODO: Keep symbolic/dynamic allocation shapes out of this reader until
  // the manifest has a finite-domain contract for them. Every current shape
  // field is a concrete positive integer and is validated redundantly below.
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
  auto schema = requiredString(object, "schema", error);
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

// Verifies one record from the filtered candidate stream. IDs are contiguous
// among packable tuples. Shape tuples themselves must remain in the
// lexicographic order produced by visitConcurrentlyPackableShapeTuples; this
// rejects duplicates and reordering without storing the entire manifest.
static bool validateCandidateAtIndex(
    uint64_t index, ArrayRef<TaskFact> tasks, ArrayRef<RectShape> shapes,
    ConcurrentPackingCache &packing, const Candidate &candidate,
    SmallVectorImpl<size_t> &previousShapeIndices, std::string &error) {
  if (candidate.id != makeSequentialCandidateId(index)) {
    error = "candidate ID does not match its canonical manifest index";
    return false;
  }
  if (candidate.choices.size() != tasks.size()) {
    error = "candidate task count does not match its declared space";
    return false;
  }
  SmallVector<size_t> shapeIndices;
  SmallVector<RectShape> selectedShapes;
  for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
    const RectShape &candidateShape = candidate.choices[taskIndex].shape;
    auto found = llvm::find_if(shapes, [&](const RectShape &legalShape) {
      return sameShape(candidateShape, legalShape);
    });
    if (found == shapes.end()) {
      error = "candidate contains a shape outside its declared shape space";
      return false;
    }
    shapeIndices.push_back(static_cast<size_t>(found - shapes.begin()));
    selectedShapes.push_back(candidateShape);
  }
  if (!packing.canPack(selectedShapes)) {
    error = "candidate task rectangles cannot fit simultaneously on the "
            "physical CGRA grid";
    return false;
  }
  if (!previousShapeIndices.empty() &&
      !std::lexicographical_compare(previousShapeIndices.begin(),
                                    previousShapeIndices.end(),
                                    shapeIndices.begin(), shapeIndices.end())) {
    error = "candidate manifest is duplicated or out of canonical order";
    return false;
  }
  previousShapeIndices.assign(shapeIndices.begin(), shapeIndices.end());
  return true;
}

// Counts the exact concurrently packable space, but stops as soon as it proves
// that the manifest's declared count is too small. Combined with strictly
// increasing legal records, equal counts prove that no valid tuple is missing.
static bool hasExactPackableCandidateCount(uint64_t declaredCount,
                                           size_t taskCount,
                                           ArrayRef<RectShape> shapes,
                                           ConcurrentPackingCache &packing) {
  uint64_t computedCount = 0;
  bool exceededDeclaredCount = false;
  bool completed = visitConcurrentlyPackableShapeTuples(
      taskCount, shapes, packing, [&](uint64_t index, ArrayRef<size_t>) {
        if (index >= declaredCount) {
          exceededDeclaredCount = true;
          return false;
        }
        computedCount = index + 1;
        return true;
      });
  return completed && !exceededDeclaredCount && computedCount == declaredCount;
}

// Compares both architecture dimensions and the exact YAML bytes. The hash
// catches capability or latency changes that dimensions alone cannot see.
static bool architectureMatches(const ManifestHeader &header,
                                const ::mlir::neura::Architecture &architecture,
                                StringRef expectedArchitectureSha256,
                                std::string &error) {
  if (header.gridRows != architecture.getMultiCgraRows() ||
      header.gridCols != architecture.getMultiCgraColumns() ||
      header.perCgraRows != architecture.getPerCgraRows() ||
      header.perCgraCols != architecture.getPerCgraColumns()) {
    error = "candidate manifest architecture dimensions do not match current "
            "Neura architecture";
    return false;
  }
  FailureOr<std::string> currentSha = currentArchitectureSha256(error);
  if (failed(currentSha))
    return false;
  if (header.architectureSha256 != *currentSha) {
    error = "candidate manifest architecture SHA-256 does not match current "
            "--architecture-spec";
    return false;
  }
  if (!expectedArchitectureSha256.empty() &&
      header.architectureSha256 != expectedArchitectureSha256) {
    error = "cost catalogue architecture SHA-256 does not match candidate "
            "manifest";
    return false;
  }
  return true;
}

// Streams and validates a complete candidate manifest before forwarding each
// candidate to the scoring or materialization callback. Every record must be a
// legal, concurrently packable shape tuple in canonical order. The footer is
// then checked against an independent traversal of the exact packable space.
bool readCandidateManifest(StringRef path, ArrayRef<TaskFact> tasks,
                           StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           StringRef expectedManifestSha256,
                           StringRef expectedArchitectureSha256,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read candidate manifest " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  const std::string manifestSha = sha256((*buffer)->getBuffer());
  if (!expectedManifestSha256.empty() &&
      manifestSha != expectedManifestSha256) {
    error = "cost catalogue candidate manifest SHA-256 does not match the "
            "candidate file";
    return false;
  }

  bool sawHeader = false;
  bool sawFooter = false;
  uint64_t count = 0;
  SmallVector<RectShape> legalShapes;
  SmallVector<size_t> previousShapeIndices;
  std::unique_ptr<ConcurrentPackingCache> packing;
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
      if (!architectureMatches(header, architecture, expectedArchitectureSha256,
                               error))
        return false;
      legalShapes = enumerateStaticRectShapes(
          header.gridRows, header.gridCols, header.perCgraRows,
          header.perCgraCols, header.maxCgrasPerTask);
      if (legalShapes.empty()) {
        error = "candidate manifest declares an empty shape space";
        return false;
      }
      packing = std::make_unique<ConcurrentPackingCache>(header.gridRows,
                                                         header.gridCols);
      sawHeader = true;
      continue;
    }
    if (*recordType == "candidate") {
      if (!sawHeader || sawFooter) {
        error = "candidate record is outside header/footer";
        return false;
      }
      Candidate candidate;
      if (!parseCandidate(*object, tasks, candidate, error) || !packing ||
          !validateCandidateAtIndex(count, tasks, legalShapes, *packing,
                                    candidate, previousShapeIndices, error) ||
          !consume(count, candidate, error))
        return false;
      ++count;
      continue;
    }
    if (*recordType == "footer") {
      auto schema = requiredString(*object, "schema", error);
      auto expectedCount = requiredInteger(*object, "candidate_count", error);
      if (!sawHeader || sawFooter || !schema || !expectedCount ||
          *schema != kCandidateSchema || *expectedCount <= 0) {
        if (error.empty())
          error = "invalid candidate manifest footer";
        return false;
      }
      if (static_cast<uint64_t>(*expectedCount) != count || !packing ||
          !hasExactPackableCandidateCount(count, tasks.size(), legalShapes,
                                          *packing)) {
        error = "candidate manifest count does not match the complete "
                "concurrently packable shape space";
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

// Loads and validates the task-shape cost catalogue. Each entry is keyed by
// task name and explicit mapper tile dimensions, so a `rect-4x8` label remains
// readable without becoming a correctness dependency.
bool TaskShapeCostCache::load(StringRef path, StringRef expectedFunction,
                              ArrayRef<TaskFact> expectedTasks,
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
  auto schema = requiredString(*root, "schema", error);
  auto function = requiredString(*root, "function", error);
  auto modelNamespace = requiredString(*root, "namespace", error);
  llvm::json::Object *metadata = root->getObject("predictor_metadata");
  llvm::json::Array *entries = root->getArray("entries");
  if (!schema || !function || !modelNamespace || !metadata || !entries)
    return false;
  if (*schema != kCostSchema || *function != expectedFunction ||
      modelNamespace->empty()) {
    error = "cost catalogue schema/function/namespace mismatch";
    return false;
  }

  // Bind the predictions to the exact frozen candidate bytes and analytical
  // inputs. A same-named task or same-sized architecture is not sufficient.
  auto candidateSha =
      requiredString(*metadata, "candidate_manifest_sha256", error);
  llvm::json::Object *provenance = metadata->getObject("analytical_provenance");
  llvm::json::Object *architectureContract =
      metadata->getObject("architecture_contract");
  llvm::json::Object *rankingPolicy = metadata->getObject("ranking_policy");
  if (!candidateSha || !provenance || !architectureContract)
    return false;
  auto architectureSha =
      requiredString(*provenance, "architecture_sha256", error);
  llvm::json::Object *taskBodyHashes =
      provenance->getObject("task_body_sha256");
  llvm::json::Object *taskHashes = provenance->getObject("task_dfg_sha256");
  if (!architectureSha || !taskBodyHashes || !taskHashes)
    return false;
  if (!isSha256(*candidateSha) || !isSha256(*architectureSha)) {
    error = "cost catalogue provenance contains an invalid SHA-256";
    return false;
  }
  auto architectureContractId =
      requiredString(*architectureContract, "contract_id", error);
  llvm::json::Array *supportedArchitectures =
      architectureContract->getArray("supported_architecture_sha256");
  if (!architectureContractId || architectureContractId->empty() ||
      !supportedArchitectures || supportedArchitectures->empty())
    return false;
  bool architectureIsSupported = false;
  for (const llvm::json::Value &value : *supportedArchitectures) {
    std::optional<StringRef> supportedSha = value.getAsString();
    if (!supportedSha || !isSha256(*supportedSha)) {
      error = "cost catalogue architecture contract contains an invalid "
              "SHA-256";
      return false;
    }
    architectureIsSupported |= *supportedSha == *architectureSha;
  }
  if (!architectureIsSupported) {
    error = "cost catalogue analytical architecture is outside the model's "
            "supported architecture contract";
    return false;
  }
  // Mapping success probability is not part of the current objective. Older
  // or diagnostic-producing adapters may still describe it, but a catalogue
  // must never request that the C++ scorer use it for rejection or ranking.
  if (rankingPolicy) {
    if (const llvm::json::Value *role =
            rankingPolicy->get("mapper_success_probability")) {
      std::optional<StringRef> value = role->getAsString();
      if (!value || *value != "diagnostic_only") {
        error = "cost catalogue may expose mapper success probability only "
                "as a diagnostic";
        return false;
      }
    }
    if (const llvm::json::Value *uses =
            rankingPolicy->get("uses_mapper_success_probability")) {
      std::optional<bool> value = uses->getAsBoolean();
      if (!value || *value) {
        error = "cost catalogue cannot use mapper success probability for "
                "scoring";
        return false;
      }
    }
  }

  if (taskBodyHashes->size() != expectedTasks.size() ||
      taskHashes->size() != expectedTasks.size()) {
    error = "cost catalogue task provenance does not exactly cover current IR "
            "tasks";
    return false;
  }
  for (const TaskFact &task : expectedTasks) {
    std::optional<StringRef> bodySha = taskBodyHashes->getString(task.name);
    std::optional<StringRef> dfgSha = taskHashes->getString(task.name);
    if (!bodySha || !dfgSha || !isSha256(*dfgSha) ||
        *bodySha != task.bodySha256) {
      error = "cost catalogue task provenance does not bind the current task "
              "body to its source DFG";
      return false;
    }
  }

  namespace_ = modelNamespace->str();
  candidateManifestSha256_ = candidateSha->str();
  architectureSha256_ = architectureSha->str();
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
      auto lowerBound = entry->getNumber("analytical_lower_bound");
      if (!ii || !startup || !lowerBound || !std::isfinite(*ii) ||
          !std::isfinite(*startup) || !std::isfinite(*lowerBound) ||
          *ii <= 0.0 || *startup <= 0.0 || *lowerBound <= 0.0 ||
          *ii < *lowerBound) {
        error = "supported cost requires positive finite predicted_ii and "
                "startup_cycles and analytical_lower_bound, and "
                "predicted_ii >= analytical_lower_bound";
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
