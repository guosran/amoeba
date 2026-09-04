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
//   enumerate -> complete candidate JSONL + unique ML cost queries
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
// Preserves exhaustive DSE by keeping scores out of enumeration, refusing to
// truncate oversized spaces, scoring every frozen record before sorting, and
// validating the complete manifest before mutating IR. This protocol version
// supports only static rectangular shapes and static trip counts.
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

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates-v2";
inline constexpr llvm::StringLiteral kCostSchema = "amoeba-task-shape-cost-v2";
inline constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-analytical-task-scores-v2";
inline constexpr llvm::StringLiteral kShapePolicy = "static-rectangles-v2";
inline constexpr llvm::StringLiteral kSearchScope = "static-shape-only-v2";
inline constexpr llvm::StringLiteral kScoreModel =
    "static-shape-compute-bottleneck-v2";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores immutable facts extracted from one Taskflow task.
struct TaskFact {
  taskflow::TaskflowTaskOp op;
  std::string name;
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

FailureOr<llvm::SmallVector<TaskFact>> collectTaskFacts(func::FuncOp func,
                                                        std::string &error);
llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols,
                                                       int64_t maxCgrasPerTask);
std::string makeSequentialCandidateId(uint64_t index);
llvm::json::Object candidateJson(const Candidate &candidate);
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error);
bool samePath(llvm::StringRef lhs, llvm::StringRef rhs);
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module,
                                           llvm::StringRef requested,
                                           std::string &error);
bool readCandidateManifest(llvm::StringRef path, llvm::ArrayRef<TaskFact> tasks,
                           llvm::StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error);

// Adapts the scorer to the ML agent's task-shape cost catalogue. Memoizes each
// (task name, mapper tile rows, mapper tile columns) lookup across the full
// candidate traversal.
class TaskShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            std::string &error);
  const TaskShapeCost *get(const TaskShapeChoice &choice, std::string &error);

  llvm::StringRef nameSpace() const { return namespace_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t entries() const { return cache_.size(); }

private:
  std::string namespace_;
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
