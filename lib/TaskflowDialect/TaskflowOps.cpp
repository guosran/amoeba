#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <cstddef>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
stateRegion(Value state, Value expectedBase) {
  if (auto join = state.getDefiningOp<TaskflowJoinOp>()) {
    if (join.getBase() != expectedBase)
      return failure();
    return std::make_pair(SmallVector<int64_t>(join.getRegionLower()),
                          SmallVector<int64_t>(join.getRegionUpper()));
  }
  auto task = state.getDefiningOp<TaskflowTaskOp>();
  if (!task || !llvm::is_contained(task.getDoneWrites(), state) ||
      task.getWillWrites().size() != 1 ||
      task.getWillWrites().front() != expectedBase)
    return failure();
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
  SmallVector<TaskflowCounterOp> counters;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
  for (auto [index, counter] : llvm::enumerate(counters)) {
    std::optional<int64_t> lb = constantIndex(counter.getLowerBound());
    std::optional<int64_t> ub = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lb || !ub || !step || *step != 1 || *ub <= *lb ||
        (index == 0 && counter.getParentIndex()) ||
        (index != 0 &&
         counter.getParentIndex() != counters[index - 1].getCounterIndex()))
      return failure();
    lower.push_back(*lb);
    upper.push_back(*ub);
  }
  if (lower.empty())
    return failure();
  return std::make_pair(std::move(lower), std::move(upper));
}

} // namespace

LogicalResult TaskflowTaskOp::verify() {
  if (!getBody().hasOneBlock())
    return emitOpError("requires exactly one body block");
  Block &body = getBody().front();
  SmallVector<Value> inputs(getWillReads());
  inputs.append(getWillWrites().begin(), getWillWrites().end());
  inputs.append(getValueInputs().begin(), getValueInputs().end());
  if (body.getNumArguments() != inputs.size())
    return emitOpError("body argument count must match execution inputs; "
                       "original memrefs are provenance only");
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (argument.getType() != input.getType())
      return emitOpError("body argument type does not match execution input");
  auto yield = dyn_cast<TaskflowYieldOp>(body.getTerminator());
  if (!yield || yield->getNumOperands() != getNumResults())
    return emitOpError("yield arity must match task results");
  for (auto [result, yielded] : llvm::zip(getResults(), yield->getOperands()))
    if (result.getType() != yielded.getType())
      return emitOpError("yield type does not match task result");
  return success();
}

LogicalResult TaskflowJoinOp::verify() {
  if (getTileStates().size() < 2)
    return emitOpError("requires at least two tile states");
  auto type = dyn_cast<MemRefType>(getBase().getType());
  if (!type || !type.hasStaticShape() || getJoined().getType() != type)
    return emitOpError(
        "requires one static memref type for base, states, and result");
  int64_t axis = getAxis();
  ArrayRef<int64_t> parentLower = getRegionLower();
  ArrayRef<int64_t> parentUpper = getRegionUpper();
  if (axis < 0 || axis >= type.getRank() ||
      parentLower.size() != static_cast<size_t>(type.getRank()) ||
      parentUpper.size() != static_cast<size_t>(type.getRank()))
    return emitOpError("has invalid axis or parent region rank");
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (parentLower[dim] < 0 || parentUpper[dim] <= parentLower[dim] ||
        parentUpper[dim] > type.getShape()[dim])
      return emitOpError("parent region is outside the static memref shape");

  SmallVector<std::pair<int64_t, int64_t>> intervals;
  llvm::SmallDenseSet<Value, 8> uniqueStates;
  for (Value state : getTileStates()) {
    if (state.getType() != type || !uniqueStates.insert(state).second)
      return emitOpError("requires distinct tile states of the base type");
    auto region = stateRegion(state, getBase());
    if (failed(region) || region->first.size() != parentLower.size() ||
        region->second.size() != parentUpper.size())
      return emitOpError("cannot prove a tile state's static region and base");
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (dim == axis)
        continue;
      if (region->first[dim] != parentLower[dim] ||
          region->second[dim] != parentUpper[dim])
        return emitOpError("tile regions differ outside the selected axis");
    }
    intervals.push_back({region->first[axis], region->second[axis]});
  }
  llvm::sort(intervals);
  int64_t cursor = parentLower[axis];
  for (auto [lower, upper] : intervals) {
    if (lower != cursor || upper <= lower)
      return emitOpError("tile regions overlap or leave a gap");
    cursor = upper;
  }
  if (cursor != parentUpper[axis])
    return emitOpError("tile regions do not exactly cover the parent region");
  return success();
}

//===----------------------------------------------------------------------===//
// TaskflowTaskOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowTaskOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parses optional task name: @Task_0.
  StringAttr task_name;
  if (succeeded(parser.parseOptionalSymbolName(task_name))) {
    result.addAttribute("task_name", task_name);
  }

  // Parses will_reads(%arg0, %arg1 : memref<?xi32>, memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  if (succeeded(parser.parseOptionalKeyword("will_reads"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses will_writes(%arg5 : memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  if (succeeded(parser.parseOptionalKeyword("will_writes"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses value_inputs: value_inputs(%scalar : i32).
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;
  if (succeeded(parser.parseOptionalKeyword("value_inputs"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  // Parses original memrefs with explicit types:
  // [original_read_memrefs(%arg0, %arg1 : type1, type2),
  // original_write_memrefs(%arg5 : type3)].
  SmallVector<OpAsmParser::UnresolvedOperand> original_read_operands;
  SmallVector<Type> original_read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> original_write_operands;
  SmallVector<Type> original_write_types;

  if (succeeded(parser.parseOptionalLSquare())) {
    // original_read_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_read_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_read_operands) ||
          parser.parseColonTypeList(original_read_types) ||
          parser.parseRParen())
        return failure();
    }

    // optional comma.
    (void)parser.parseOptionalComma();

    // original_write_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_write_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_write_operands) ||
          parser.parseColonTypeList(original_write_types) ||
          parser.parseRParen())
        return failure();
    }

    if (parser.parseRSquare())
      return failure();
  }

  // Validates operand/type count match.
  if (read_operands.size() != read_types.size() ||
      write_operands.size() != write_types.size() ||
      value_operands.size() != value_types.size() ||
      original_read_operands.size() != original_read_types.size() ||
      original_write_operands.size() != original_write_types.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "operand and type count mismatch");
  }

  // Resolves all operands.
  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_read_operands, original_read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_write_operands, original_write_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // Parses optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parses function type: : (...) -> (...).
  FunctionType func_type;
  if (parser.parseColon() || parser.parseType(func_type))
    return failure();

  // Adds result types.
  result.addTypes(func_type.getResults());

  // Parses region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*args=*/{}, /*argTypes=*/{})) {
    return failure();
  }

  // Adds operand segment sizes.
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(read_operands.size()),
           static_cast<int32_t>(write_operands.size()),
           static_cast<int32_t>(value_operands.size()),
           static_cast<int32_t>(original_read_operands.size()),
           static_cast<int32_t>(original_write_operands.size())}));

  // Adds result segment sizes. Read/write outputs are inferred from the
  // terminator, because multiple will_writes states may map to a single
  // done_writes state.
  size_t num_read_outputs = 0;
  size_t num_write_outputs = 0;
  if (!body->empty()) {
    if (auto yield_op =
            dyn_cast<TaskflowYieldOp>(body->front().getTerminator())) {
      num_read_outputs = yield_op.getDoneReads().size();
      num_write_outputs = yield_op.getDoneWrites().size();
    }
  }

  size_t num_value_outputs = 0;
  size_t total_memref_results = 0;
  for (Type t : func_type.getResults()) {
    if (isa<MemRefType>(t)) {
      total_memref_results++;
    } else {
      num_value_outputs++;
    }
  }
  if (total_memref_results != num_read_outputs + num_write_outputs) {
    return parser.emitError(parser.getCurrentLocation(),
                            "taskflow.yield memref result count does not "
                            "match task function result type");
  }
  result.addAttribute("resultSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(num_read_outputs),
                           static_cast<int32_t>(num_write_outputs),
                           static_cast<int32_t>(num_value_outputs)}));

  return success();
}

void TaskflowTaskOp::print(OpAsmPrinter &printer) {
  // Prints task name.
  printer << " @" << getTaskName();

  // Prints will_reads.
  if (!getWillReads().empty()) {
    printer << " will_reads(";
    llvm::interleaveComma(getWillReads(), printer);
    printer << " : ";
    llvm::interleaveComma(getWillReads().getTypes(), printer);
    printer << ")";
  }

  // Prints will_writes.
  if (!getWillWrites().empty()) {
    printer << " will_writes(";
    llvm::interleaveComma(getWillWrites(), printer);
    printer << " : ";
    llvm::interleaveComma(getWillWrites().getTypes(), printer);
    printer << ")";
  }

  // Prints value_inputs.
  if (!getValueInputs().empty()) {
    printer << " value_inputs(";
    llvm::interleaveComma(getValueInputs(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueInputs().getTypes(), printer);
    printer << ")";
  }

  // Prints original memrefs with types.
  if (!getOriginalReadMemrefs().empty() || !getOriginalWriteMemrefs().empty()) {
    printer << " [";

    if (!getOriginalReadMemrefs().empty()) {
      printer << "original_read_memrefs(";
      llvm::interleaveComma(getOriginalReadMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalReadMemrefs().getTypes(), printer);
      printer << ")";
    }

    if (!getOriginalReadMemrefs().empty() && !getOriginalWriteMemrefs().empty())
      printer << ", ";

    if (!getOriginalWriteMemrefs().empty()) {
      printer << "original_write_memrefs(";
      llvm::interleaveComma(getOriginalWriteMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalWriteMemrefs().getTypes(), printer);
      printer << ")";
    }

    printer << "]";
  }

  // Prints attributes (skip operandSegmentSizes, resultSegmentSizes,
  // task_name).
  SmallVector<StringRef> elidedAttrs = {"operandSegmentSizes",
                                        "resultSegmentSizes", "task_name"};
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Prints function type.
  printer << " : (";
  llvm::interleaveComma(llvm::concat<const Type>(getWillReads().getTypes(),
                                                 getWillWrites().getTypes(),
                                                 getValueInputs().getTypes()),
                        printer);
  printer << ") -> (";
  llvm::interleaveComma(llvm::concat<const Type>(getDoneReads().getTypes(),
                                                 getDoneWrites().getTypes(),
                                                 getValueOutputs().getTypes()),
                        printer);
  printer << ")";

  // Prints region.
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/true);
}

//===----------------------------------------------------------------------===//
// TaskflowYieldOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowYieldOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;

  // Parses done_reads (WAR dependency passthrough).
  if (succeeded(parser.parseOptionalKeyword("done_reads"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses done_writes.
  if (succeeded(parser.parseOptionalKeyword("done_writes"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses values.
  if (succeeded(parser.parseOptionalKeyword("values"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(read_operands.size()),
                           static_cast<int32_t>(write_operands.size()),
                           static_cast<int32_t>(value_operands.size())}));

  return success();
}

void TaskflowYieldOp::print(OpAsmPrinter &printer) {
  if (!getDoneReads().empty()) {
    printer << " done_reads(";
    llvm::interleaveComma(getDoneReads(), printer);
    printer << " : ";
    llvm::interleaveComma(getDoneReads().getTypes(), printer);
    printer << ")";
  }

  if (!getDoneWrites().empty()) {
    printer << " done_writes(";
    llvm::interleaveComma(getDoneWrites(), printer);
    printer << " : ";
    llvm::interleaveComma(getDoneWrites().getTypes(), printer);
    printer << ")";
  }

  if (!getValueResults().empty()) {
    printer << " values(";
    llvm::interleaveComma(getValueResults(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueResults().getTypes(), printer);
    printer << ")";
  }
}
