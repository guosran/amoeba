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
// validating the complete manifest before mutating IR. Uses SHA-256 only for
// stable content identity and tamper/staleness detection, never as a score.
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
#include "llvm/Support/SHA256.h"
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
    "amoeba-analytical-task-candidates-v1";
inline constexpr llvm::StringLiteral kCostSchema = "amoeba-task-shape-cost-v1";
inline constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-analytical-task-scores-v1";
inline constexpr llvm::StringLiteral kShapePolicy = "rectangles-v1";
inline constexpr llvm::StringLiteral kSearchScope = "shape-only-v1";
inline constexpr llvm::StringLiteral kCandidateIdentity =
    "shape-candidate-sha256-v1";
inline constexpr llvm::StringLiteral kScoreModel =
    "shape-only-compute-bottleneck-v1";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;
  std::string mapperShapeId;

  int64_t cgraCount() const { return rows * cols; }
  std::string irAttr() const;
};

// Stores immutable facts extracted from one Taskflow task.
struct TaskFact {
  taskflow::TaskflowTaskOp op;
  std::string name;
  std::string bodyId;
  int64_t tripCount = 1;
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  std::string bodyId;
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
  std::string architectureFingerprint;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  int64_t perCgraRows = 0;
  int64_t perCgraCols = 0;
  int64_t maxCgrasPerTask = 0;
};

// Stores the record count and ordered-ID digest that close a manifest.
struct ManifestFooter {
  uint64_t candidateCount = 0;
  std::string candidateIdsSha256;
};

// Stores the ML prediction for one task-body and mapper-shape query.
struct BodyShapeCost {
  double predictedII = 0.0;
  double startupCycles = 0.0;
  bool supported = false;
};

using CostKey = std::tuple<std::string, std::string, std::string>;

// Stores the sortable score for one supported program candidate.
struct RankedCandidate {
  std::string id;
  double score = 0.0;
};

using CandidateConsumer =
    llvm::function_ref<bool(uint64_t, const Candidate &, std::string &)>;

// Computes a lowercase SHA-256 digest for stable content identity.
std::string sha256Hex(llvm::StringRef text);
// Adds one candidate ID and a record separator to the manifest digest.
void updateCandidateIdDigest(llvm::SHA256 &hasher, llvm::StringRef candidateId);
FailureOr<std::string> currentArchitectureFingerprint(std::string &error);
FailureOr<llvm::SmallVector<TaskFact>> collectTaskFacts(func::FuncOp func,
                                                        std::string &error);
llvm::SmallVector<RectShape>
enumerateRectShapes(int64_t gridRows, int64_t gridCols, int64_t perCgraRows,
                    int64_t perCgraCols, int64_t maxCgrasPerTask);
std::string makeCandidateId(llvm::StringRef function,
                            llvm::StringRef architectureFingerprint,
                            llvm::ArrayRef<TaskShapeChoice> choices);
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
                           llvm::StringRef architectureFingerprint,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error);

// Adapts the scorer to the ML agent's task-shape cost catalogue. Memoizes each
// (task, body ID, mapper shape) lookup across the full candidate traversal.
class BodyShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            llvm::StringRef expectedArchitectureFingerprint,
            std::string &error);
  const BodyShapeCost *get(const TaskShapeChoice &choice, std::string &error);

  llvm::StringRef nameSpace() const { return namespace_; }
  llvm::StringRef catalogSha256() const { return catalogSha256_; }
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

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_DSE_SUPPORT_H
