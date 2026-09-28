#!/usr/bin/env python3
"""Materialize a semantic ORBIT GEMM variant as Taskflow MLIR.

The semantic variant enumerator is the owner of tiling, regions, and edges.
This module is deliberately a small, source-derived printer: it consumes a
canonical program identity and one already legalized variant and emits the
Taskflow dialect form used by the existing graph-facts pass.  It does not
consult a backend, an architecture, a cost model, or a schedule.

The generated task bodies use the canonical static ``taskflow.counter`` /
``taskflow.hyperblock`` shape.  Besides making the IR directly verifiable,
that shape keeps the materialization consumable by the existing ORBIT graph
facts extraction pass.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any, Iterable, Mapping, Sequence


class TaskflowMaterializationError(ValueError):
    """Raised when a semantic program/variant cannot be materialized."""


def _as_identity(program: Any) -> Mapping[str, Any]:
    if isinstance(program, Mapping):
        identity = program.get("program_identity", program)
        if isinstance(identity, Mapping):
            return identity
    raise TaskflowMaterializationError(
        "materialize_taskflow requires a canonical program identity mapping")


def _identity_fields(program: Any) -> dict[str, Any]:
    source = _as_identity(program)
    required = (
        "canonical_program_id", "source_program_sha256", "function", "M",
        "N", "K", "dtype", "layout", "C_initialization", "alias_proof",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise TaskflowMaterializationError(
            f"canonical program identity is missing {missing!r}")
    for key in ("M", "N", "K"):
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TaskflowMaterializationError(
                f"canonical program identity {key} must be positive")
    if not isinstance(source["dtype"], str) or not source["dtype"]:
        raise TaskflowMaterializationError("canonical program dtype is invalid")
    if source["layout"] not in ("row_major", "row_major_identity"):
        raise TaskflowMaterializationError("unsupported canonical program layout")
    return dict(source)


def _variant_fields(variant: Any) -> Mapping[str, Any]:
    if not isinstance(variant, Mapping):
        raise TaskflowMaterializationError("semantic variant must be a mapping")
    for key in ("variant_id", "canonical_program_id", "tasks", "edges",
                "output_tiles", "k_blocks", "k_policy", "BM", "BN",
                "BK", "tail_policy"):
        if key not in variant:
            raise TaskflowMaterializationError(f"variant.{key} is missing")
    if variant.get("semantic_legal") is not True:
        raise TaskflowMaterializationError("variant is not semantically legal")
    if not isinstance(variant["tasks"], list) or not variant["tasks"]:
        raise TaskflowMaterializationError("variant.tasks must be non-empty")
    if not isinstance(variant["edges"], list):
        raise TaskflowMaterializationError("variant.edges must be an array")
    return variant


def _mlir_string(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    return json.dumps(value, ensure_ascii=True)


def _mlir_array(values: Sequence[int]) -> str:
    return "array<i64: " + ", ".join(str(int(value)) for value in values) + ">"


def _quote_identifier(value: str, fallback: str = "value") -> str:
    # MLIR SSA names and symbols are deliberately kept boring.  The semantic
    # IDs remain in attributes; this is only the printable local spelling.
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not result:
        result = fallback
    if result[0].isdigit():
        result = "_" + result
    return result


def _ssa(value: str) -> str:
    return "%" + _quote_identifier(value)


def _region_buffer(region: Any) -> str | None:
    if not isinstance(region, Mapping):
        return None
    value = region.get("buffer")
    return value if isinstance(value, str) else None


def _region_range(region: Any) -> tuple[list[int], list[int]] | None:
    if not isinstance(region, Mapping):
        return None
    lower, upper = region.get("lower"), region.get("upper")
    if (not isinstance(lower, list) or not isinstance(upper, list) or
            len(lower) != len(upper) or
            any(isinstance(v, bool) or not isinstance(v, int)
                for v in lower + upper)):
        raise TaskflowMaterializationError("malformed semantic region")
    return list(lower), list(upper)


def _task_regions(task: Mapping[str, Any]) -> Mapping[str, Any]:
    regions = task.get("regions")
    if not isinstance(regions, Mapping):
        raise TaskflowMaterializationError(
            f"task {task.get('task_id')!r} has no grouped regions")
    return regions


def _task_id(task: Mapping[str, Any]) -> str:
    value = task.get("task_id")
    if not isinstance(value, str) or not value:
        raise TaskflowMaterializationError("every task needs a string task_id")
    return value


def _task_kind(task: Mapping[str, Any]) -> str:
    value = task.get("task_kind", task.get("task_role", "gemm"))
    if value == "compute":
        return "gemm"
    return str(value)


def _edge_endpoints(variant: Mapping[str, Any]) -> tuple[
        dict[str, list[Mapping[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    incoming: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_ids = {_task_id(task) for task in variant["tasks"]}
    for edge in variant["edges"]:
        if not isinstance(edge, Mapping):
            raise TaskflowMaterializationError("variant edge must be an object")
        source, target = edge.get("source_task"), edge.get("target_task")
        if source not in task_ids or target not in task_ids or source == target:
            raise TaskflowMaterializationError(
                f"edge has invalid endpoints {source!r} -> {target!r}")
        incoming[target].append(edge)
        outgoing[source].append(edge)
    for values in incoming.values():
        values.sort(key=lambda edge: (
            str(edge.get("kind", "")), str(edge.get("source_task", ""))))
    for values in outgoing.values():
        values.sort(key=lambda edge: (
            str(edge.get("kind", "")), str(edge.get("target_task", ""))))
    return incoming, outgoing


def _topological_tasks(variant: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks = list(variant["tasks"])
    by_id = {_task_id(task): task for task in tasks}
    if len(by_id) != len(tasks):
        raise TaskflowMaterializationError("variant task IDs are not unique")
    incoming, outgoing = _edge_endpoints(variant)
    indegree = {task_id: len(incoming.get(task_id, []))
                for task_id in by_id}
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    ready.sort()
    result: list[Mapping[str, Any]] = []
    while ready:
        current = ready.pop(0)
        result.append(by_id[current])
        for edge in outgoing.get(current, []):
            target = str(edge["target_task"])
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(result) != len(tasks):
        raise TaskflowMaterializationError("semantic task graph contains a cycle")
    return result


def _all_private_buffers(tasks: Iterable[Mapping[str, Any]],
                         k_policy: str) -> list[str]:
    buffers: set[str] = set()
    for task in tasks:
        regions = _task_regions(task)
        candidates: list[Any] = list(regions.values())
        candidates.extend(regions.get("partial_Cs", []) or [])
        for region in candidates:
            buffer = _region_buffer(region)
            if buffer is not None and buffer not in {"A", "B", "C"}:
                buffers.add(buffer)
        for key in ("partial_lifetime", "state_lifetime"):
            lifetime = task.get(key)
            if isinstance(lifetime, Mapping):
                buffer = lifetime.get("buffer")
                if isinstance(buffer, str) and buffer not in {"A", "B", "C"}:
                    buffers.add(buffer)
        # Intermediate sequential K states are materialized as private
        # memrefs.  The variant's C_output region remains the logical C tile;
        # this extra state is the SSA/storage realization of that logical
        # carry, and the final K task alone writes the external C base.
        if (k_policy == "sequential_accumulate" and
                _task_kind(task) == "gemm" and
                task.get("final_C_ownership") != "external_final_C"):
            buffers.add("state_C/" + _task_id(task))
    return sorted(buffers)


def _attr_lines(identity: Mapping[str, Any], variant: Mapping[str, Any],
                indent: str = "  ") -> list[str]:
    values: list[tuple[str, str]] = [
        ("amoeba.semantic.schema", _mlir_string(variant.get("schema", ""))),
        ("amoeba.semantic.canonical_program_id",
         _mlir_string(identity["canonical_program_id"])),
        ("amoeba.semantic.variant_id", _mlir_string(variant["variant_id"])),
        ("amoeba.semantic.source_program_sha256",
         _mlir_string(identity["source_program_sha256"])),
        ("amoeba.semantic.operation_access_sha256",
         _mlir_string(identity.get("operation_access_sha256", ""))),
        ("amoeba.semantic.source_function", _mlir_string(identity["function"])),
        ("amoeba.semantic.layout", _mlir_string(identity["layout"])),
        ("amoeba.semantic.C_initialization",
         _mlir_string(identity["C_initialization"])),
        ("amoeba.semantic.alias_proof", _mlir_string(identity["alias_proof"])),
        ("amoeba.semantic.BM", f"{int(variant['BM'])} : i64"),
        ("amoeba.semantic.BN", f"{int(variant['BN'])} : i64"),
        ("amoeba.semantic.BK", f"{int(variant['BK'])} : i64"),
        ("amoeba.semantic.tail_policy", _mlir_string(variant["tail_policy"])),
        ("amoeba.semantic.k_policy", _mlir_string(variant["k_policy"])),
        ("amoeba.semantic.reduction_topology",
         _mlir_string(variant.get("reduction_topology", "none"))),
        ("amoeba.semantic.arithmetic_semantics",
         _mlir_string(variant.get("arithmetic_semantics", ""))),
        ("amoeba.semantic.completion_task_id",
         _mlir_string(variant.get("completion_task_id", "completion_join"))),
        ("amoeba.semantic.provenance",
         _mlir_string("canonical_source_to_taskflow")),
    ]
    return [indent + key + " = " + value + "," for key, value in values]


def _task_attr_lines(identity: Mapping[str, Any], variant: Mapping[str, Any],
                     task: Mapping[str, Any], incoming: Sequence[Mapping[str, Any]],
                     outgoing: Sequence[Mapping[str, Any]],
                     indent: str = "         ") -> list[str]:
    kind = _task_kind(task)
    role = "compute" if kind == "gemm" else str(kind)
    lines = [
        indent + "amoeba.semantic.task_id = " + _mlir_string(_task_id(task)) + ",",
        indent + "amoeba.semantic.task_role = " + _mlir_string(role) + ",",
        indent + "amoeba.semantic.task_kind = " + _mlir_string(kind) + ",",
        indent + "amoeba.semantic.variant_id = " +
        _mlir_string(variant["variant_id"]) + ",",
        indent + "amoeba.semantic.canonical_program_id = " +
        _mlir_string(identity["canonical_program_id"]) + ",",
        indent + "amoeba.semantic.source_program_sha256 = " +
        _mlir_string(identity["source_program_sha256"]) + ",",
        indent + "amoeba.semantic.logical_output_tile_id = " +
        _mlir_string(task.get("logical_output_tile_id", "")) + ",",
    ]
    if task.get("logical_k_block_id") is not None:
        lines.append(indent + "amoeba.semantic.logical_k_block_id = " +
                     _mlir_string(task["logical_k_block_id"]) + ",")
    for name, key in (("M", "m_range"), ("N", "n_range"),
                      ("K", "k_range")):
        value = task.get(key)
        if isinstance(value, list):
            lines.append(indent + f"amoeba.semantic.{name}_range = " +
                         _mlir_array(value) + ",")
    regions = _task_regions(task)
    for name in ("A_read", "B_read", "C_input", "C_output", "partial_C"):
        region = regions.get(name)
        bounds = _region_range(region)
        if bounds is None:
            continue
        lines.append(indent + f"amoeba.semantic.{name}_region = " +
                     _mlir_string(json.dumps(
                         region, sort_keys=True, separators=(",", ":"))) + ",")
        lower, upper = bounds
        lines.append(indent + f"amoeba.semantic.{name}_lower = " +
                     _mlir_array(lower) + ",")
        lines.append(indent + f"amoeba.semantic.{name}_upper = " +
                     _mlir_array(upper) + ",")
        buffer = _region_buffer(region)
        if buffer is not None:
            lines.append(indent + f"amoeba.semantic.{name}_buffer = " +
                         _mlir_string(buffer) + ",")
    partials = regions.get("partial_Cs")
    if isinstance(partials, list):
        lines.append(indent + "amoeba.semantic.partial_C_count = " +
                     f"{len(partials)} : i64,")
    if isinstance(task.get("reduction_topology"), str):
        lines.append(indent + "amoeba.semantic.reduction_topology = " +
                     _mlir_string(task["reduction_topology"]) + ",")
    for field in ("reduction_order", "reduction_input_task_ids"):
        value = task.get(field)
        if isinstance(value, list):
            lines.append(indent + f"amoeba.semantic.{field} = [" + ", ".join(
                _mlir_string(item) for item in value) + "],")
    for field in ("initial_C_ownership", "final_C_ownership",
                  "reduction_associativity"):
        if field in task:
            lines.append(indent + f"amoeba.semantic.{field} = " +
                         _mlir_string(task[field]) + ",")
    for field in ("partial_lifetime", "state_lifetime"):
        value = task.get(field)
        if isinstance(value, Mapping):
            lines.append(indent + f"amoeba.semantic.{field} = " +
                         _mlir_string(json.dumps(
                             value, sort_keys=True, separators=(",", ":"))) + ",")
    predecessor_ids = sorted(str(edge["source_task"]) for edge in incoming)
    successor_ids = sorted(str(edge["target_task"]) for edge in outgoing)
    incoming_edges = sorted(
        f"{edge['source_task']}|{edge.get('kind', '')}|{edge.get('scope', '')}"
        for edge in incoming)
    outgoing_edges = sorted(
        f"{edge['target_task']}|{edge.get('kind', '')}|{edge.get('scope', '')}"
        for edge in outgoing)
    lines.append(indent + "amoeba.semantic.predecessors = [" + ", ".join(
        _mlir_string(value) for value in predecessor_ids) + "],")
    lines.append(indent + "amoeba.semantic.successors = [" + ", ".join(
        _mlir_string(value) for value in successor_ids) + "],")
    lines.append(indent + "amoeba.semantic.incoming_edges = [" + ", ".join(
        _mlir_string(value) for value in incoming_edges) + "],")
    lines.append(indent + "amoeba.semantic.outgoing_edges = [" + ", ".join(
        _mlir_string(value) for value in outgoing_edges) + "],")
    return lines


def _constants_and_counters(ranges: Sequence[tuple[int, int]],
                            prefix: str) -> tuple[list[str], list[str], list[str]]:
    values = {0, 1}
    for lower, upper in ranges:
        values.add(lower)
        values.add(upper)
    constants = [f"%c{value} = arith.constant {value} : index"
                 for value in sorted(values)]
    counters: list[str] = []
    indices: list[str] = []
    previous: str | None = None
    for ordinal, (lower, upper) in enumerate(ranges):
        index = f"%{prefix}{ordinal}"
        indices.append(index)
        parent = (f" parent({previous} : index)" if previous is not None else "")
        counters.append(
            f"{index} = taskflow.counter{parent} from %c{lower} to "
            f"%c{upper} step %c1 : index")
        previous = index
    return constants, counters, indices


def _arith_names(dtype: str) -> tuple[str, str, str]:
    if dtype == "i32":
        return "i32", "muli", "addi"
    if dtype == "f32":
        return "f32", "mulf", "addf"
    raise TaskflowMaterializationError(
        f"semantic Taskflow materializer does not support dtype {dtype!r}")


def _memref_type(rows: int, cols: int, dtype: str) -> str:
    return f"memref<{rows}x{cols}x{dtype}>"


def _task_body_compute(identity: Mapping[str, Any], task: Mapping[str, Any],
                       read_count: int, write_count: int,
                       dtype: str) -> list[str]:
    m0, m1 = task["m_range"]
    n0, n1 = task["n_range"]
    k0, k1 = task["k_range"]
    constants, counters, indices = _constants_and_counters(
        [(m0, m1), (n0, n1), (k0, k1)], "i")
    lines = ["      " + line for line in constants + counters]
    lines.append(
        '      "taskflow.hyperblock"(' + ", ".join(indices) +
        ') <{operandSegmentSizes = array<i32: 3, 0>}> ({')
    lines.append("      ^bb0(%mi: index, %ni: index, %ki: index):")
    element_type, multiply, add = _arith_names(dtype)
    # The body arguments are will-reads followed by will-writes.  A GEMM
    # compute task always has A and B as its first two read states.
    lines.extend([
        f"        %a = memref.load %arg0[%mi, %ki] : "
        f"{_memref_type(identity['M'], identity['K'], dtype)}",
        f"        %b = memref.load %arg1[%ki, %ni] : "
        f"{_memref_type(identity['K'], identity['N'], dtype)}",
        f"        %product = arith.{multiply} %a, %b : {element_type}",
    ])
    if task.get("C_output_region") is not None:
        # Sequential K: the third read state is the prior C state and the
        # first write state is the same logical state.  The external C state
        # is therefore read once by the first K task and carried explicitly.
        lines.extend([
            f"        %c = memref.load %arg2[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}",
            f"        %updated = arith.{add} %c, %product : {element_type}",
            f"        memref.store %updated, %arg{read_count}[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}",
        ])
    else:
        # Parallel K: partial C was zero-filled before task execution.  Every
        # iteration reads and updates only its own private partial buffer.
        lines.extend([
            f"        %partial = memref.load %arg{read_count}[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}",
            f"        %updated = arith.{add} %partial, %product : {element_type}",
            f"        memref.store %updated, %arg{read_count}[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}",
        ])
    lines.extend([
        "        taskflow.hyperblock.yield",
        "      }) : (index, index, index) -> ()",
        f"      taskflow.yield done_writes(%arg{read_count} : "
        f"{_memref_type(identity['M'], identity['N'], dtype)})",
    ])
    return lines


def _task_body_reduction(identity: Mapping[str, Any], task: Mapping[str, Any],
                         input_count: int, dtype: str) -> list[str]:
    m0, m1 = task["m_range"]
    n0, n1 = task["n_range"]
    constants, counters, indices = _constants_and_counters(
        [(m0, m1), (n0, n1)], "i")
    lines = ["      " + line for line in constants + counters]
    lines.append(
        '      "taskflow.hyperblock"(' + ", ".join(indices) +
        ') <{operandSegmentSizes = array<i32: 2, 0>}> ({')
    lines.append("      ^bb0(%mi: index, %ni: index):")
    element_type, _, add = _arith_names(dtype)
    regions = _task_regions(task)
    direct_regions = regions.get("partial_Cs", [])
    if not isinstance(direct_regions, list) or not direct_regions:
        raise TaskflowMaterializationError(
            f"reduction task {_task_id(task)} has no direct partial regions")
    direct_count = len(direct_regions)
    # Inputs are direct reduction operands first; initial C, when present, is
    # appended after them by the task printer.  In the body, C is placed first
    # in the arithmetic expression so C is included exactly once.
    c_input = regions.get("C_input")
    has_initial_c = _region_buffer(c_input) == "C"
    arg_indices = list(range(direct_count))
    lines_for_loads: list[str] = []
    for ordinal in arg_indices:
        lines_for_loads.append(
            f"        %input{ordinal} = memref.load %arg{ordinal}[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}")
    lines.extend(lines_for_loads)
    if has_initial_c:
        c_arg = direct_count
        lines.append(
            f"        %initial = memref.load %arg{c_arg}[%mi, %ni] : "
            f"{_memref_type(identity['M'], identity['N'], dtype)}")
        accumulator = "%initial"
        start = 0
    else:
        accumulator = "%input0"
        start = 1
    for ordinal in range(start, direct_count):
        result = f"%sum{ordinal}"
        lines.append(
            f"        {result} = arith.{add} {accumulator}, %input{ordinal} : "
            f"{element_type}")
        accumulator = result
    # A no-C reduction with one input is a legal tree carry only when the
    # semantic variant explicitly asks for it; no arithmetic op is needed.
    lines.extend([
        f"        memref.store {accumulator}, %arg{input_count}[%mi, %ni] : "
        f"{_memref_type(identity['M'], identity['N'], dtype)}",
        "        taskflow.hyperblock.yield",
        "      }) : (index, index) -> ()",
        f"      taskflow.yield done_writes(%arg{input_count} : "
        f"{_memref_type(identity['M'], identity['N'], dtype)})",
    ])
    return lines


def _task_output_region(task: Mapping[str, Any]) -> tuple[list[int], list[int]] | None:
    regions = _task_regions(task)
    return _region_range(regions.get("C_output"))


def _tiling_contract_attrs(input_regions: Sequence[Any],
                           output_regions: Sequence[Any]) -> list[str]:
    """Print the per-operand region contract consumed by TaskEdgeContract.

    The semantic region facts are logical half-open regions.  TaskEdgeContract
    expects one descriptor for each ``will_reads`` and ``will_writes`` entry,
    in the same order as the corresponding original memory operands.  Keeping
    these arrays aligned is what lets the C++ graph builder prove that final
    C writes for different output tiles are disjoint.
    """

    def bounds(regions: Sequence[Any]) -> tuple[list[str], list[str]]:
        lowers: list[str] = []
        uppers: list[str] = []
        for region in regions:
            pair = _region_range(region)
            if pair is None:
                raise TaskflowMaterializationError(
                    "TaskEdgeContract requires a known region for every memory operand")
            lower, upper = pair
            lowers.append(_mlir_array(lower))
            uppers.append(_mlir_array(upper))
        return lowers, uppers

    input_lowers, input_uppers = bounds(input_regions)
    output_lowers, output_uppers = bounds(output_regions)
    lines = [
        "         amoeba.tiling.input_region_lowers = [" +
        ", ".join(input_lowers) + "],",
        "         amoeba.tiling.input_region_uppers = [" +
        ", ".join(input_uppers) + "],",
        "         amoeba.tiling.input_region_reasons = [" +
        ", ".join(_mlir_string("proven_rectangular")
                   for _ in input_regions) + "],",
        # The Taskflow join verifier consumes these as a one-output proof;
        # TaskEdgeContract consumes the lower/upper arrays by write index.
        "         amoeba.tiling.output_axes = array<i64: " +
        ", ".join("0" for _ in output_regions) + ">,",
        "         amoeba.tiling.output_region_lowers = [" +
        ", ".join(output_lowers) + "],",
        "         amoeba.tiling.output_region_uppers = [" +
        ", ".join(output_uppers) + "],",
    ]
    return lines


def _task_operation(identity: Mapping[str, Any], variant: Mapping[str, Any],
                    task: Mapping[str, Any], incoming: Sequence[Mapping[str, Any]],
                    outgoing: Sequence[Mapping[str, Any]],
                    values: Mapping[str, str], allocations: Mapping[str, str],
                    dtype: str) -> tuple[list[str], str]:
    task_id = _task_id(task)
    kind = _task_kind(task)
    if kind not in ("gemm", "reduction"):
        raise TaskflowMaterializationError(
            f"cannot materialize semantic task kind {kind!r}")
    type_a = _memref_type(identity["M"], identity["K"], dtype)
    type_b = _memref_type(identity["K"], identity["N"], dtype)
    type_c = _memref_type(identity["M"], identity["N"], dtype)
    reads: list[str] = []
    writes: list[str] = []
    original_reads: list[str] = []
    original_writes: list[str] = []
    input_regions: list[Any] = []
    output_regions: list[Any] = []
    if kind == "gemm":
        a_value, b_value = "%A", "%B"
        reads.extend((a_value, b_value))
        original_reads.extend((a_value, b_value))
        regions = _task_regions(task)
        input_regions.extend((regions.get("A_read"), regions.get("B_read")))
        if variant["k_policy"] == "sequential_accumulate":
            c_region = regions.get("C_input")
            # The semantic region keeps the logical output buffer name C for
            # every K block; ownership, rather than that spelling, tells us
            # whether this block consumes the external initial state or the
            # preceding task's explicit state result.
            if task.get("initial_C_ownership") != "external_initial_C":
                predecessor = next((edge for edge in incoming
                                    if edge.get("kind") == "state_carry"), None)
                if predecessor is None or predecessor["source_task"] not in values:
                    raise TaskflowMaterializationError(
                        f"sequential task {task_id} has no C state predecessor")
                c_state = values[str(predecessor["source_task"])]
            else:
                c_state = "%C"
            reads.append(c_state)
            input_regions.append(c_region)
            # Taskflow's write state remains explicitly bound to the
            # original C base only for the final K block.  Intermediate
            # blocks publish private C states; their SSA results form the
            # explicit carry without creating concurrent final-C writers.
            if task.get("final_C_ownership") == "external_final_C":
                writes.append("%C")
            else:
                private_state = "state_C/" + task_id
                if private_state not in allocations:
                    raise TaskflowMaterializationError(
                        f"missing private sequential state {private_state}")
                writes.append(allocations[private_state])
            original_reads.append("%C")
            original_writes.append("%C" if writes[-1] == "%C" else writes[-1])
            output_regions.append(regions.get("C_output"))
        else:
            partial_region = regions.get("partial_C")
            partial = _region_buffer(partial_region)
            if partial is None or partial not in allocations:
                raise TaskflowMaterializationError(
                    f"parallel task {task_id} has no private partial buffer")
            partial_value = allocations[partial]
            writes.append(partial_value)
            original_writes.append(partial_value)
            output_regions.append(partial_region)
    else:
        regions = _task_regions(task)
        direct_ids = task.get("reduction_input_task_ids")
        if not isinstance(direct_ids, list):
            direct_ids = [edge["source_task"] for edge in incoming
                          if edge.get("kind") in ("reduction_input", "state_carry")]
        if not direct_ids:
            raise TaskflowMaterializationError(
                f"reduction task {task_id} has no direct input tasks")
        for source in direct_ids:
            source = str(source)
            if source not in values:
                raise TaskflowMaterializationError(
                    f"reduction task {task_id} references unavailable {source}")
            reads.append(values[source])
            original_reads.append(values[source])
        direct_regions = regions.get("partial_Cs")
        if not isinstance(direct_regions, list) or len(direct_regions) != len(direct_ids):
            raise TaskflowMaterializationError(
                f"reduction task {task_id} has unaligned input regions")
        input_regions.extend(direct_regions)
        c_input = regions.get("C_input")
        if _region_buffer(c_input) == "C":
            reads.append("%C")
            original_reads.append("%C")
            input_regions.append(c_input)
        output = _region_buffer(regions.get("C_output"))
        if output == "C":
            output_value = "%C"
            original_writes.append("%C")
        elif output is not None and output in allocations:
            output_value = allocations[output]
            original_writes.append(output_value)
        else:
            raise TaskflowMaterializationError(
                f"reduction task {task_id} has no output state")
        writes.append(output_value)
        output_regions.append(regions.get("C_output"))

    operands = reads + writes
    if not operands:
        raise TaskflowMaterializationError(f"task {task_id} has no operands")
    # Keep the type list tied to each operand instead of deriving it from the
    # source buffer spelling: A/B/C and every private state are all static
    # memref<MxNxi32> except the two read-only GEMM inputs.
    if kind == "gemm":
        operand_types = ([type_a, type_b] +
                         [type_c] * (len(reads) - 2) +
                         [type_c] * len(writes))
    else:
        operand_types = [type_c] * (len(reads) + len(writes))
    if len(operand_types) != len(operands):
        raise TaskflowMaterializationError(
            f"internal operand type mismatch for task {task_id}")
    def joined(values_: Sequence[str]) -> str:
        return ", ".join(values_)
    def types(values_: Sequence[str]) -> str:
        return ", ".join(values_)
    lines: list[str] = []
    result_name = _ssa(task_id)
    lines.append(f"    {result_name} = taskflow.task @{_quote_identifier(task_id)}")
    lines.append(f"        will_reads({joined(reads)} : {types(operand_types[:len(reads)])})")
    lines.append(f"        will_writes({joined(writes)} : "
                 f"{types(operand_types[len(reads):])})")
    lines.append("        [original_read_memrefs(" + joined(original_reads) +
                 " : " + ", ".join(type_a if value == "%A" else
                                      type_b if value == "%B" else type_c
                                      for value in original_reads) + "),")
    lines.append("         original_write_memrefs(" + joined(original_writes) +
                 " : " + ", ".join(type_c for _ in original_writes) + ")]" )
    attr_lines = _task_attr_lines(identity, variant, task, incoming, outgoing)
    attr_lines.extend(_tiling_contract_attrs(input_regions, output_regions))
    output_region = _task_output_region(task)
    actual_output_buffer = "C" if writes and writes[-1] == "%C" else (
        _region_buffer(_task_regions(task).get("C_output")) if kind == "reduction"
        else ("state_C/" + task_id if variant["k_policy"] == "sequential_accumulate"
              else _region_buffer(_task_regions(task).get("partial_C"))))
    attr_lines.append("         amoeba.semantic.materialized_output_buffer = " +
                      _mlir_string(actual_output_buffer or "") + ",")
    if output_region is not None and actual_output_buffer == "C":
        lower, upper = output_region
        attr_lines.extend([
            "         amoeba.tiling.output_region_lower = " +
            _mlir_array(lower) + ",",
            "         amoeba.tiling.output_region_upper = " +
            _mlir_array(upper) + ",",
        ])
    # Remove the trailing comma from the final attribute in the dictionary.
    lines.append("        {")
    lines.extend(attr_lines[:-1] + ([attr_lines[-1].rstrip(",")] if attr_lines else []))
    lines.append("        }")
    lines.append("        : (" + ", ".join(operand_types) + ") -> (" + type_c + ") {")
    lines.append("      ^bb0(" + ", ".join(
        f"%arg{index}: {type_}" for index, type_ in enumerate(operand_types)) + "):")
    if kind == "gemm":
        lines.extend(_task_body_compute(identity, task, len(reads), len(writes), dtype))
    else:
        lines.extend(_task_body_reduction(identity, task, len(reads), dtype))
    lines.append("    }")
    return lines, result_name


def _join_operation(name: str, states: Sequence[str], base: str, axis: int,
                    lower: Sequence[int], upper: Sequence[int],
                    identity: Mapping[str, Any], variant: Mapping[str, Any],
                    region_kind: str) -> list[str]:
    if len(states) < 2:
        raise TaskflowMaterializationError("a Taskflow join needs two states")
    type_c = _memref_type(identity["M"], identity["N"], identity["dtype"])
    attrs = [
        "amoeba.semantic.task_role = " + _mlir_string("completion_only"),
        "amoeba.semantic.join_kind = " + _mlir_string(region_kind),
        "amoeba.semantic.variant_id = " + _mlir_string(variant["variant_id"]),
        "amoeba.semantic.canonical_program_id = " +
        _mlir_string(identity["canonical_program_id"]),
    ]
    return [
        f"    %join_{_quote_identifier(name)} = taskflow.join states(" +
        ", ".join(states) + f") base({base}) axis({axis}) region(" +
        "[" + ", ".join(str(int(value)) for value in lower) + "], [" +
        ", ".join(str(int(value)) for value in upper) + "]) {" +
        ", ".join(attrs) + "} : " + type_c,
    ]


def _completion_states(identity: Mapping[str, Any], variant: Mapping[str, Any],
                       output_values: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Build hierarchical joins for a 2-D tile grid.

    ``taskflow.join`` proves intervals along one axis.  A rectangular MxN
    grid therefore needs one N-axis join per M row followed by one M-axis
    join.  Every join is a completion token; none performs arithmetic.
    """
    tiles = variant["output_tiles"]
    groups: dict[tuple[int, int], list[tuple[list[int], list[int], str]]] = defaultdict(list)
    for tile in tiles:
        tile_id = str(tile["logical_output_tile_id"])
        if tile_id not in output_values:
            raise TaskflowMaterializationError(f"missing final state for {tile_id}")
        groups[(int(tile["m_range"][0]), int(tile["m_range"][1]))].append(
            (list(tile["m_range"]), list(tile["n_range"]), output_values[tile_id]))
    rows: list[tuple[list[int], str]] = []
    join_lines: list[str] = []
    for row_index, key in enumerate(sorted(groups)):
        entries = sorted(groups[key], key=lambda item: item[1])
        if len(entries) == 1:
            rows.append((entries[0][0], entries[0][2]))
            continue
        m_range = entries[0][0]
        states = [entry[2] for entry in entries]
        n_end = max(entry[1][1] for entry in entries)
        name = f"row_{m_range[0]}_{m_range[1]}"
        join_lines.extend(_join_operation(name, states, "%C", 1,
                                          [m_range[0], 0],
                                          [m_range[1], n_end], identity,
                                          variant, "completion_only"))
        rows.append((m_range, f"%join_{_quote_identifier(name)}"))
    if len(rows) == 1:
        return join_lines, [rows[0][1]]
    states = [state for _, state in rows]
    m_end = max(region[1] for region, _ in rows)
    join_lines.extend(_join_operation("all_output_tiles", states, "%C", 0,
                                      [0, 0], [m_end, identity["N"]], identity,
                                      variant, "completion_only"))
    return join_lines, [f"%join_{_quote_identifier('all_output_tiles')}"]


def materialize_taskflow(program: Any, variant: Mapping[str, Any]) -> str:
    """Return valid Taskflow MLIR for one semantic variant.

    ``program`` is the canonical identity returned by
    ``canonical_program_identity``.  Every task, region, and dependency is
    taken from ``variant``; the only derived details are SSA plumbing, static
    counter syntax, and the hierarchical completion joins required by the
    Taskflow dialect verifier.
    """

    identity = _identity_fields(program)
    variant = _variant_fields(variant)
    if variant["canonical_program_id"] != identity["canonical_program_id"]:
        raise TaskflowMaterializationError(
            "variant and canonical program identities do not match")
    # The updated semantic space includes this field.  Older legal variants
    # default to the only topology that is meaningful for sequential K.
    topology = variant.get("reduction_topology", "none")
    if variant["k_policy"] == "parallel_partial_reduce" and topology not in (
            "linear", "balanced_tree"):
        raise TaskflowMaterializationError("parallel variant has no reduction topology")
    if variant["k_policy"] == "sequential_accumulate" and topology != "none":
        raise TaskflowMaterializationError(
            "sequential variant cannot carry a reduction topology")
    dtype, _, _ = _arith_names(str(identity["dtype"]))
    tasks = _topological_tasks(variant)
    incoming, outgoing = _edge_endpoints(variant)
    completion_id = variant.get("completion_task_id")
    compute_tasks = [task for task in tasks if _task_id(task) != completion_id]
    private_buffers = _all_private_buffers(compute_tasks, variant["k_policy"])
    allocations = {buffer: "%alloc_" + _quote_identifier(buffer)
                   for buffer in private_buffers}
    type_c = _memref_type(identity["M"], identity["N"], dtype)
    lines: list[str] = [
        "module {",
        f"  func.func @{_quote_identifier(str(identity['function']))}(",
        f"      %A: {_memref_type(identity['M'], identity['K'], dtype)} "
        "{amoeba.noalias},",
        f"      %B: {_memref_type(identity['K'], identity['N'], dtype)} "
        "{amoeba.noalias},",
        f"      %C: {type_c} {{amoeba.noalias}}) -> {type_c} attributes {{",
    ]
    function_attrs = _attr_lines(identity, variant)
    lines.extend(function_attrs[:-1] + ([function_attrs[-1].rstrip(",")] if function_attrs else []))
    lines.append("  } {")
    if private_buffers:
        lines.append("    %zero = arith.constant " +
                     ("0.0" if dtype == "f32" else "0") + " : " + dtype)
        for buffer in private_buffers:
            lines.append(f"    {allocations[buffer]} = memref.alloc() : {type_c}")
            lines.append(
                f"    linalg.fill ins(%zero : {dtype}) "
                f"outs({allocations[buffer]} : {type_c})")
    values: dict[str, str] = {}
    for task in compute_tasks:
        task_id = _task_id(task)
        task_lines, result = _task_operation(
            identity, variant, task, incoming.get(task_id, []),
            outgoing.get(task_id, []), values, allocations, dtype)
        lines.extend(task_lines)
        values[task_id] = result
    final_values: dict[str, str] = {}
    for tile in variant["output_tiles"]:
        tile_id = str(tile["logical_output_tile_id"])
        candidates = [task for task in compute_tasks
                      if task.get("logical_output_tile_id") == tile_id]
        if not candidates:
            raise TaskflowMaterializationError(f"output tile {tile_id} has no tasks")
        terminal = [task for task in candidates
                    if not any(edge.get("source_task") == task["task_id"] and
                               edge.get("target_task") in {
                                   candidate["task_id"] for candidate in candidates}
                               for edge in outgoing.get(task["task_id"], []))]
        if len(terminal) != 1:
            raise TaskflowMaterializationError(
                f"output tile {tile_id} has {len(terminal)} terminal tasks")
        final_values[tile_id] = values[_task_id(terminal[0])]
    join_lines, completion_values = _completion_states(identity, variant, final_values)
    lines.extend(join_lines)
    if len(completion_values) != 1:
        raise TaskflowMaterializationError("completion materialization produced no token")
    lines.append(f"    return {completion_values[0]} : {type_c}")
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


__all__ = ["TaskflowMaterializationError", "materialize_taskflow"]
