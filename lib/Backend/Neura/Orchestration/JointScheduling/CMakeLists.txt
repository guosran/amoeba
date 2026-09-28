get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraJointScheduling
  AnalyticalTaskCandidateCommon.cpp
  AnalyticalTaskCandidateManifest.cpp
  AnalyticalTaskCostCatalog.cpp
  AnalyzeRecResMiiPass.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  EnumerateJointSemanticRewritePass.cpp
  MaterializeJointSemanticRewritePass.cpp
  JointReductionRewritePasses.cpp
  ExtractJointTaskGraphFactsPass.cpp
  MaterializeAnalyticalTaskCandidatePass.cpp
  MapJointSchedulingTasksPass.cpp
  ScoreAnalyticalTaskCandidatesPass.cpp
  SpatialTaskCandidateSpace.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraOrchestrationUtil
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRNeura
  MLIRNeuraTransforms
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTaskflow
  ${dialect_libs}
  LLVMSupport
)

add_executable(amoeba-joint-scheduling-runtime-contract-test
  ${PROJECT_SOURCE_DIR}/test/Backend/Neura/JointSchedulingRuntimeContractTest.cpp
)

target_link_libraries(amoeba-joint-scheduling-runtime-contract-test PRIVATE
  MLIRAmoebaNeuraJointScheduling
  MLIRAmoebaNeuraOrchestrationUtil
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTaskflow
  LLVMSupport
)
