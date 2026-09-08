//===- AnalyticalTaskDSESupport.h - Shared task DSE support -----*- C++ -*-===//
//
// Defines the internal records and file protocol shared by the analytical
// task-DSE passes. Keeps this header private to the optimization library.
//
// Coordinates the shape-only flow through three separately invocable passes:
//
//   Taskflow IR + architecture.yaml
//              |
//              v
//   enumerate -> every concurrently packable shape tuple + ML cost queries
//              |
//              v
//   score     -> one score per candidate + deterministic top-k shortlist
//              |
//              v
//   materialize one shortlist entry -> unchanged heuristic mapper pipeline
//
// Distinguishes a physical shape, measured in CGRA instances, from its mapper
// shape, measured in tiles. Derives the conversion from the architecture YAML;
// for a 4x4-tile CGRA, physical 1x2 therefore becomes mapper shape 4x8.
//
// A candidate fixes one oriented rectangle per task. It is retained only when
// those rectangles have at least one simultaneous, non-overlapping placement
// on the physical multi-CGRA grid. The concrete placement is still chosen by
// the existing downstream heuristic; temporal reuse cannot make an
// over-capacity shape tuple legal in this shape-only stage.
//
// TODO: Replace that temporary simultaneous-residency rule with analytical
// spatial-temporal scheduling. For example, this stage currently rejects two
// 4x4 tasks on a 4x4 grid, although a temporal schedule could run the second
// task after the first one releases the grid.
//
// Preserves exhaustive DSE by keeping scores out of enumeration, refusing to
// truncate oversized spaces, scoring every frozen record before sorting, and
// validating the complete concurrently packable manifest before mutating IR.
// This protocol supports only static rectangular shapes and static trip
// counts.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_DSE_SUPPORT_H
#define AMOEBA_ANALYTICAL_TASK_DSE_SUPPORT_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates";
inline constexpr llvm::StringLiteral kCostSchema = "amoeba-task-shape-cost";
inline constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-analytical-task-scores";
inline constexpr llvm::StringLiteral kShapePolicy =
    "static-oriented-rectangles";
inline constexpr llvm::StringLiteral kSearchScope =
    "static-shape-concurrent-fit";
inline constexpr llvm::StringLiteral kSpatialCapacityPolicy =
    "all-tasks-simultaneous-exact-pack";
inline constexpr llvm::StringLiteral kScoreModel =
    "static-shape-compute-bottleneck";
inline constexpr llvm::StringLiteral kSourceTaskBodyShaAttr =
    "amoeba.source_task_body_sha256";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores immutable identity and trip-count facts from one Taskflow task.
struct TaskFact {
  taskflow::TaskflowTaskOp op;
  std::string name;
  std::string bodySha256;
  int64_t tripCount = 1;
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  int64_t tripCount = 1;
  RectShape shape;
};

// Stores one ordered shape choice for every Taskflow task.
struct Candidate {
  std::string id;
  llvm::SmallVector<TaskShapeChoice> choices;
};

// Stores the values that define a finite candidate space.
struct ManifestHeader {
  std::string function;
  std::string architectureSha256;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  int64_t perCgraRows = 0;
  int64_t perCgraCols = 0;
  int64_t maxCgrasPerTask = 0;
};

// Stores the record count that closes a manifest.
struct ManifestFooter {
  uint64_t candidateCount = 0;
};

// Stores the ML prediction for one task and mapper-shape query.
struct TaskShapeCost {
  double predictedII = 0.0;
  double startupCycles = 0.0;
  bool supported = false;
};

using CostKey = std::tuple<std::string, int64_t, int64_t>;

// Stores the sortable score for one supported program candidate.
struct RankedCandidate {
  std::string id;
  uint64_t manifestIndex = 0;
  double score = 0.0;
};

using CandidateConsumer =
    llvm::function_ref<bool(uint64_t, const Candidate &, std::string &)>;
using ShapeIndexTupleConsumer =
    llvm::function_ref<bool(uint64_t, llvm::ArrayRef<size_t>)>;

// Memoizes exact physical-grid packing by the sorted multiset of oriented
// rectangles. Task names and task order do not affect whether the rectangles
// fit, so many ordered candidates share one small backtracking result.
class ConcurrentPackingCache {
public:
  ConcurrentPackingCache(int64_t gridRows, int64_t gridCols)
      : gridRows_(gridRows), gridCols_(gridCols) {}

  bool canPack(llvm::ArrayRef<RectShape> shapes);
  int64_t gridRows() const { return gridRows_; }
  int64_t gridCols() const { return gridCols_; }

private:
  using Key = std::vector<std::pair<int64_t, int64_t>>;

  int64_t gridRows_ = 0;
  int64_t gridCols_ = 0;
  std::map<Key, bool> results_;
};

FailureOr<llvm::SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error);
llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols,
                                                       int64_t maxCgrasPerTask);
// Visits every shape tuple that admits a simultaneous, non-overlapping
// placement on the physical grid. `shapeIndices` follows task order and indexes
// `shapes`; the valid candidate index is contiguous and starts at zero. Returns
// false only when the consumer requests an early stop.
bool visitConcurrentlyPackableShapeTuples(size_t taskCount,
                                          llvm::ArrayRef<RectShape> shapes,
                                          ConcurrentPackingCache &packing,
                                          ShapeIndexTupleConsumer consume);
std::string makeSequentialCandidateId(uint64_t index);
llvm::json::Object candidateJson(const Candidate &candidate);
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error);
bool samePath(llvm::StringRef lhs, llvm::StringRef rhs);
FailureOr<std::string> currentArchitectureSha256(std::string &error);
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module,
                                           llvm::StringRef requested,
                                           std::string &error);
bool readCandidateManifest(llvm::StringRef path, llvm::ArrayRef<TaskFact> tasks,
                           llvm::StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           llvm::StringRef expectedManifestSha256,
                           llvm::StringRef expectedArchitectureSha256,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error);

// Adapts the scorer to the ML agent's task-shape cost catalogue. Memoizes each
// (task name, mapper tile rows, mapper tile columns) lookup across the full
// candidate traversal.
class TaskShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            llvm::ArrayRef<TaskFact> expectedTasks, std::string &error);
  const TaskShapeCost *get(const TaskShapeChoice &choice, std::string &error);

  llvm::StringRef nameSpace() const { return namespace_; }
  llvm::StringRef candidateManifestSha256() const {
    return candidateManifestSha256_;
  }
  llvm::StringRef architectureSha256() const { return architectureSha256_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t entries() const { return cache_.size(); }
  uint64_t catalogEntries() const { return catalog_.size(); }

private:
  std::string namespace_;
  std::string candidateManifestSha256_;
  std::string architectureSha256_;
  std::map<CostKey, TaskShapeCost> catalog_;
  std::map<CostKey, TaskShapeCost> cache_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_DSE_SUPPORT_H
