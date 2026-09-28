#!/usr/bin/env python3
"""Reference materializer for source-bound joint rewrite graphs.

The C++ MLIR passes own production rewrite legality and materialization.
This module is a reference printer: it consumes a legalized test graph,
preserves its task/edge regions, and emits verifier-ready Taskflow IR.  The
printer has no architecture, mapping, cost, or replay inputs.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "orbit-joint-taskflow-materialization-v1"
REGION_SCHEMA = "orbit-half-open-region-v1"


class TaskflowMaterializationError(ValueError):
    """A graph could not be materialized without weakening its proof."""


def _rewrite_module():
    import orbit_joint_rewrite_space
    return orbit_joint_rewrite_space


def _graph(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("graph"), Mapping) \
            and "tasks" not in value:
        value = value["graph"]
    if not isinstance(value, Mapping):
        raise TaskflowMaterializationError("joint graph must be a mapping")
    return value


def _quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _ident(value: Any, fallback: str = "value") -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not text:
        text = fallback
    if text[0].isdigit():
        text = "_" + text
    return text


def _ssa(value: Any) -> str:
    return "%" + _ident(value)


def _array(values: Sequence[int]) -> str:
    return "array<i64: " + ", ".join(str(int(v)) for v in values) + ">"


def _region(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskflowMaterializationError("semantic operand is missing a region")
    if value.get("schema") != REGION_SCHEMA or value.get("half_open") is not True:
        raise TaskflowMaterializationError("semantic region is not explicit half-open form")
    lower, upper = value.get("lower"), value.get("upper")
    if (not isinstance(lower, list) or not isinstance(upper, list) or
            len(lower) != len(upper) or
            any(isinstance(v, bool) or not isinstance(v, int) for v in lower + upper)):
        raise TaskflowMaterializationError("semantic region bounds are malformed")
    return value


def _bounds(value: Any) -> tuple[list[int], list[int]]:
    region = _region(value)
    return list(region["lower"]), list(region["upper"])


def _buf(value: Any) -> str:
    region = _region(value)
    buffer = region.get("buffer")
    if not isinstance(buffer, str) or not buffer:
        raise TaskflowMaterializationError("semantic region has no buffer identity")
    return buffer


def _task_id(task: Mapping[str, Any]) -> str:
    value = task.get("task_id")
    if not isinstance(value, str) or not value:
        raise TaskflowMaterializationError("every graph task needs a task_id")
    return value


def _task_kind(task: Mapping[str, Any]) -> str:
    return str(task.get("task_kind", task.get("task_role", "")))


def _dimensions(graph: Mapping[str, Any]) -> tuple[int, int, int]:
    dims = graph.get("dimensions")
    if not isinstance(dims, Mapping):
        raise TaskflowMaterializationError("graph dimensions are missing")
    values = tuple(dims.get(key) for key in ("M", "N", "K"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise TaskflowMaterializationError("graph dimensions are invalid")
    return values  # type: ignore[return-value]


def _identity(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    value = graph.get("program_identity", graph.get("source_identity"))
    if not isinstance(value, Mapping):
        raise TaskflowMaterializationError("graph has no source identity")
    required = ("function", "M", "N", "K", "dtype", "layout")
    missing = [key for key in required if key not in value]
    if missing:
        raise TaskflowMaterializationError(
            f"source identity misses {missing!r}")
    if value.get("dtype") != "i32":
        raise TaskflowMaterializationError("only i32 modulo arithmetic is materialized")
    if value.get("layout") not in ("row_major", "row_major_identity"):
        raise TaskflowMaterializationError("source layout is not row-major identity")
    return value


def _edges(graph: Mapping[str, Any]) -> tuple[dict[str, list[Mapping[str, Any]]],
                                                 dict[str, list[Mapping[str, Any]]]]:
    incoming: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_ids = {_task_id(task) for task in graph.get("tasks", [])
                if isinstance(task, Mapping)}
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            raise TaskflowMaterializationError("graph edge is not a mapping")
        source, target = edge.get("source_task"), edge.get("target_task")
        if source not in task_ids or target not in task_ids or source == target:
            raise TaskflowMaterializationError("graph edge has invalid endpoints")
        incoming[str(target)].append(edge)
        outgoing[str(source)].append(edge)
    for values in incoming.values():
        values.sort(key=lambda edge: (str(edge.get("kind", "")),
                                      str(edge.get("source_task", ""))))
    for values in outgoing.values():
        values.sort(key=lambda edge: (str(edge.get("kind", "")),
                                      str(edge.get("target_task", ""))))
    return incoming, outgoing


def _topological(graph: Mapping[str, Any], incoming: Mapping[str, Sequence[Mapping[str, Any]]],
                 outgoing: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    tasks = [task for task in graph.get("tasks", [])
             if isinstance(task, Mapping) and _task_kind(task) != "completion_join"]
    by_id = {_task_id(task): task for task in tasks}
    indegree = {task_id: len([edge for edge in incoming.get(task_id, [])
                              if edge.get("source_task") in by_id])
                for task_id in by_id}
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    result: list[Mapping[str, Any]] = []
    while ready:
        current = ready.pop(0)
        result.append(by_id[current])
        for edge in outgoing.get(current, []):
            target = str(edge.get("target_task"))
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(result) != len(tasks):
        raise TaskflowMaterializationError("joint graph contains a cycle")
    return result


def _memref(rows: int, cols: int) -> str:
    return f"memref<{rows}x{cols}xi32>"


def _region_for_task(task: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    regions = task.get("regions")
    if not isinstance(regions, Mapping):
        raise TaskflowMaterializationError(f"task {_task_id(task)} has no regions")
    return _region(regions.get(key))


def _tile_region(task: Mapping[str, Any], buffer: str, access: str = "read") -> dict[str, Any]:
    m_range, n_range = task.get("m_range"), task.get("n_range")
    if not isinstance(m_range, list) or not isinstance(n_range, list):
        raise TaskflowMaterializationError(f"task {_task_id(task)} has no output range")
    return {"schema": REGION_SCHEMA, "buffer": buffer, "lower": [m_range[0], n_range[0]],
            "upper": [m_range[1], n_range[1]], "half_open": True, "access": access}


def _task_attrs(graph: Mapping[str, Any], task: Mapping[str, Any],
                in_edges: Sequence[Mapping[str, Any]],
                out_edges: Sequence[Mapping[str, Any]],
                inputs: Sequence[Any], outputs: Sequence[Any]) -> list[str]:
    graph_id = graph.get("graph_id", "")
    attrs = [
        "amoeba.semantic.schema = " + _quote(SCHEMA),
        "amoeba.semantic.graph_id = " + _quote(graph_id),
        "amoeba.semantic.canonical_source_id = " + _quote(graph.get("canonical_source_id", "")),
        "amoeba.semantic.source_provenance = " + _quote("canonical_source_to_joint_rewrite"),
        "amoeba.semantic.task_id = " + _quote(_task_id(task)),
        "amoeba.semantic.task_kind = " + _quote(_task_kind(task)),
        "amoeba.semantic.task_role = " + _quote(task.get("task_role", _task_kind(task))),
        "amoeba.semantic.intermediate_visibility = " +
        _quote(task.get("intermediate_visibility", graph.get("intermediate", {}).get("visibility", ""))),
        "amoeba.semantic.k_policy = " + _quote(task.get("k_policy", graph.get("k_policy", ""))),
        "amoeba.semantic.reduction_topology = " +
        _quote(task.get("reduction_topology", graph.get("reduction_topology", "none"))),
    ]
    for name, key in (("M", "m_range"), ("N", "n_range"), ("K", "k_range")):
        value = task.get(key)
        if isinstance(value, list):
            attrs.append(f"amoeba.semantic.{name}_range = {_array(value)}")
    if task.get("initial_C_ownership") is not None:
        attrs.append("amoeba.semantic.initial_C_ownership = " +
                     _quote(task["initial_C_ownership"]))
    if task.get("final_C_ownership") is not None:
        attrs.append("amoeba.semantic.final_C_ownership = " +
                     _quote(task["final_C_ownership"]))
    if _task_kind(task) in ("reduction", "fused_reduction_consumer"):
        attrs.append("amoeba.semantic.numeric_reduction = true")
    if _task_kind(task) == "fused_producer_consumer":
        attrs.append("amoeba.semantic.fusion_layer = " + _quote("task_graph_fusion"))
    if _task_kind(task) == "fused_reduction_consumer":
        attrs.append("amoeba.semantic.fusion_layer = " + _quote("task_graph_fusion"))
    edge_strings = []
    for edge in list(in_edges) + list(out_edges):
        # The C++ graph-facts consumer treats this compact triple as the
        # source-bound edge contract.  Regions remain independently attached
        # to the task operands below.
        if edge.get("source_task") == _task_id(task):
            edge_strings.append("{}|{}|{}".format(
                edge.get("target_task"), edge.get("kind"), edge.get("scope")))
        else:
            edge_strings.append("{}|{}|{}".format(
                edge.get("source_task"), edge.get("kind"), edge.get("scope")))
    attrs.append("amoeba.semantic.edge_annotations = [" + ", ".join(
        _quote(item) for item in edge_strings) + "]")

    # The graph-facts pass validates incoming annotations against the actual
    # SSA Taskflow edges.  Keep this contract separate from the compact,
    # bidirectional diagnostic list above: an incoming entry names the source
    # task, its semantic role, and the exact proven scope.
    incoming_strings = [
        "{}|{}|{}".format(
            edge.get("source_task"), edge.get("role", edge.get("kind", "")),
            edge.get("scope", ""))
        for edge in in_edges
    ]
    outgoing_strings = [
        "{}|{}|{}".format(
            edge.get("target_task"), edge.get("role", edge.get("kind", "")),
            edge.get("scope", ""))
        for edge in out_edges
    ]
    attrs.append("amoeba.semantic.incoming_edges = [" + ", ".join(
        _quote(item) for item in incoming_strings) + "]")
    attrs.append("amoeba.semantic.outgoing_edges = [" + ", ".join(
        _quote(item) for item in outgoing_strings) + "]")

    def add_regions(prefix: str, regions: Sequence[Any]) -> None:
        lowers, uppers, reasons = [], [], []
        for value in regions:
            lower, upper = _bounds(value)
            lowers.append(_array(lower))
            uppers.append(_array(upper))
            reasons.append(_quote("source_bound_exact_region"))
        attrs.extend([
            f"amoeba.tiling.{prefix}_region_lowers = [" + ", ".join(lowers) + "]",
            f"amoeba.tiling.{prefix}_region_uppers = [" + ", ".join(uppers) + "]",
            f"amoeba.tiling.{prefix}_region_reasons = [" + ", ".join(reasons) + "]",
        ])
    add_regions("input", inputs)
    add_regions("output", outputs)
    # Taskflow's join verifier uses the plural region contract whenever it is
    # present.  One axis descriptor is required for each will_writes result;
    # output regions here are rank-2 M/N tiles, so axis zero is the stable
    # descriptor used by the source-bound graph facts contract.
    if outputs:
        attrs.append("amoeba.tiling.output_axes = array<i64: " +
                     ", ".join("0" for _ in outputs) + ">")
    # Taskflow's completion-join verifier consumes the singular output
    # descriptor from each terminal task to recover its exact tile region;
    # the plural arrays above remain the per-operand C++ graph-facts contract.
    if len(outputs) == 1:
        lower, upper = _bounds(outputs[0])
        attrs.extend([
            "amoeba.tiling.output_region_lower = " + _array(lower),
            "amoeba.tiling.output_region_upper = " + _array(upper),
        ])
    return attrs


def _constants_and_counters(m_range: Sequence[int], n_range: Sequence[int],
                            k_range: Sequence[int] | None = None) -> tuple[list[str], list[str], list[str]]:
    ranges = [m_range, n_range]
    values = {0, 1}
    for lower, upper in ranges:
        values.update((int(lower), int(upper)))
    if k_range is not None:
        values.update((int(k_range[0]), int(k_range[1])))
    constants = [f"%c{value} = arith.constant {value} : index" for value in sorted(values)]
    counters: list[str] = []
    indices: list[str] = []
    previous: str | None = None
    for ordinal, (lower, upper) in enumerate(ranges):
        index = f"%i{ordinal}"
        parent = f" parent({previous} : index)" if previous else ""
        counters.append(f"{index} = taskflow.counter{parent} from %c{lower} to %c{upper} step %c1 : index")
        indices.append(index)
        previous = index
    return constants, counters, indices


def _scf_reduction(lines: list[str], *, m_index: str, n_index: str,
                   k_range: Sequence[int], a_arg: int, b_arg: int,
                   init_expr: str, write_arg: int, fused: bool,
                   e_arg: int | None, type_c: str) -> None:
    k0, k1 = int(k_range[0]), int(k_range[1])
    lines.append(f"        %sum = scf.for %ki = %c{k0} to %c{k1} step %c1")
    lines.append(f"            iter_args(%acc = {init_expr}) -> (i32) {{")
    lines.extend([
        f"          %a = memref.load %arg{a_arg}[{m_index}, %ki] : {type_c.replace('8x8', '8x8')}",
        f"          %b = memref.load %arg{b_arg}[%ki, {n_index}] : {type_c}",
        "          %product = arith.muli %a, %b : i32",
        "          %updated = arith.addi %acc, %product : i32",
        "          scf.yield %updated : i32",
        "        }",
    ])
    value = "%sum"
    if fused:
        if e_arg is None:
            raise TaskflowMaterializationError("fused task has no E operand")
        lines.extend([
            f"        %e = memref.load %arg{e_arg}[{m_index}, {n_index}] : {type_c}",
            "        %final = arith.addi %sum, %e : i32",
        ])
        value = "%final"
    lines.append(f"        memref.store {value}, %arg{write_arg}[{m_index}, {n_index}] : {type_c}")


def _body_gemm(task: Mapping[str, Any], *, read_count: int, write_arg: int,
               parallel: bool, fused: bool, e_arg: int | None, type_c: str) -> list[str]:
    m_range, n_range = task["m_range"], task["n_range"]
    k_range = task.get("k_range")
    if not isinstance(k_range, list):
        raise TaskflowMaterializationError(f"GEMM task {_task_id(task)} has no K range")
    constants, counters, indices = _constants_and_counters(m_range, n_range, k_range)
    lines = ["      " + item for item in constants + counters]
    lines.append('      "taskflow.hyperblock"(' + ", ".join(indices) +
                 ') <{operandSegmentSizes = array<i32: 2, 0>}> ({')
    lines.append("      ^bb0(%mi: index, %ni: index):")
    if parallel:
        lines.append("        %zero = arith.constant 0 : i32")
        init = "%zero"
        a_arg, b_arg = 0, 1
    else:
        c_arg = 2
        lines.append(f"        %initial = memref.load %arg{c_arg}[%mi, %ni] : {type_c}")
        init = "%initial"
        a_arg, b_arg = 0, 1
    _scf_reduction(lines, m_index="%mi", n_index="%ni", k_range=k_range,
                   a_arg=a_arg, b_arg=b_arg, init_expr=init,
                   write_arg=write_arg, fused=fused, e_arg=e_arg,
                   type_c=type_c)
    lines.extend([
        "        taskflow.hyperblock.yield",
        "      }) : (index, index) -> ()",
        f"      taskflow.yield done_writes(%arg{write_arg} : {type_c})",
    ])
    return lines


def _body_consumer(task: Mapping[str, Any], *, write_arg: int, type_c: str) -> list[str]:
    constants, counters, indices = _constants_and_counters(task["m_range"], task["n_range"])
    lines = ["      " + item for item in constants + counters]
    lines.append('      "taskflow.hyperblock"(' + ", ".join(indices) +
                 ') <{operandSegmentSizes = array<i32: 2, 0>}> ({')
    lines.extend([
        "      ^bb0(%mi: index, %ni: index):",
        f"        %t = memref.load %arg0[%mi, %ni] : {type_c}",
        f"        %e = memref.load %arg1[%mi, %ni] : {type_c}",
        "        %sum = arith.addi %t, %e : i32",
        f"        memref.store %sum, %arg{write_arg}[%mi, %ni] : {type_c}",
        "        taskflow.hyperblock.yield",
        "      }) : (index, index) -> ()",
        f"      taskflow.yield done_writes(%arg{write_arg} : {type_c})",
    ])
    return lines


def _tree_reduce(lines: list[str], count: int, type_c: str) -> str:
    values = [f"%input{index}" for index in range(count)]
    if not values:
        raise TaskflowMaterializationError("reduction has no partial inputs")
    level = 0
    while len(values) > 1:
        next_values: list[str] = []
        for index in range(0, len(values), 2):
            if index + 1 == len(values):
                next_values.append(values[index])
                continue
            result = f"%tree_{level}_{index // 2}"
            lines.append(f"        {result} = arith.addi {values[index]}, {values[index + 1]} : i32")
            next_values.append(result)
        values = next_values
        level += 1
    return values[0]


def _body_reduction(task: Mapping[str, Any], *, partial_count: int,
                    write_arg: int, fused: bool, e_arg: int | None,
                    type_c: str) -> list[str]:
    constants, counters, indices = _constants_and_counters(task["m_range"], task["n_range"])
    lines = ["      " + item for item in constants + counters]
    lines.append('      "taskflow.hyperblock"(' + ", ".join(indices) +
                 ') <{operandSegmentSizes = array<i32: 2, 0>}> ({')
    lines.append("      ^bb0(%mi: index, %ni: index):")
    for index in range(partial_count):
        lines.append(f"        %input{index} = memref.load %arg{index}[%mi, %ni] : {type_c}")
    c_arg = partial_count
    lines.append(f"        %initial = memref.load %arg{c_arg}[%mi, %ni] : {type_c}")
    topology = str(task.get("reduction_topology", "linear"))
    if topology == "balanced_tree":
        reduced = _tree_reduce(lines, partial_count, type_c)
    else:
        reduced = "%input0"
        for index in range(1, partial_count):
            result = f"%linear_{index}"
            lines.append(f"        {result} = arith.addi {reduced}, %input{index} : i32")
            reduced = result
    lines.append(f"        %with_c = arith.addi %initial, {reduced} : i32")
    final = "%with_c"
    if fused:
        if e_arg is None:
            raise TaskflowMaterializationError("fused reduction has no E operand")
        lines.append(f"        %e = memref.load %arg{e_arg}[%mi, %ni] : {type_c}")
        lines.append("        %final = arith.addi %with_c, %e : i32")
        final = "%final"
    lines.extend([
        f"        memref.store {final}, %arg{write_arg}[%mi, %ni] : {type_c}",
        "        taskflow.hyperblock.yield",
        "      }) : (index, index) -> ()",
        f"      taskflow.yield done_writes(%arg{write_arg} : {type_c})",
    ])
    return lines


def _task_regions_for_operation(graph: Mapping[str, Any], task: Mapping[str, Any],
                                inputs: Sequence[str], outputs: Sequence[str]) -> tuple[list[Any], list[Any]]:
    kind = _task_kind(task)
    regions = task.get("regions")
    if not isinstance(regions, Mapping):
        raise TaskflowMaterializationError(f"task {_task_id(task)} has no regions")
    if kind == "gemm_producer":
        input_regions = [_region_for_task(task, "A_read"), _region_for_task(task, "B_read")]
        if task.get("k_policy") == "sequential":
            input_regions.append(_region_for_task(task, "C_input"))
        output_key = "partial_output" if task.get("k_policy") == "parallel" else "T_output"
        output_regions = [_region_for_task(task, output_key)]
    elif kind == "elementwise_consumer":
        input_regions = [_region_for_task(task, "T_read"), _region_for_task(task, "E_read")]
        output_regions = [_region_for_task(task, "D_output")]
    elif kind == "fused_producer_consumer":
        input_regions = [_region_for_task(task, "A_read"), _region_for_task(task, "B_read")]
        input_regions.append(_region_for_task(task, "C_input"))
        input_regions.append(_tile_region(task, "E", "read"))
        output_regions = [_tile_region(task, "D", "write")]
    elif kind in ("reduction", "fused_reduction_consumer"):
        partials = regions.get("partial_inputs")
        if not isinstance(partials, list) or not partials:
            raise TaskflowMaterializationError(f"reduction task {_task_id(task)} has no partial regions")
        input_regions = [_region(item) for item in partials]
        input_regions.append(_region_for_task(task, "C_initial_input"))
        if kind == "fused_reduction_consumer":
            input_regions.append(_tile_region(task, "E", "read"))
        output_regions = [_region_for_task(task, "T_output")]
    else:
        raise TaskflowMaterializationError(f"unsupported materialized task kind {kind!r}")
    if len(inputs) != len(input_regions) or len(outputs) != len(output_regions):
        raise TaskflowMaterializationError(
            f"task {_task_id(task)} has unaligned operand regions")
    return input_regions, output_regions


def _join_line(name: str, states: Sequence[str], lower: Sequence[int], upper: Sequence[int],
               axis: int, graph: Mapping[str, Any], type_c: str, kind: str = "completion_only") -> str:
    if len(states) < 2:
        raise TaskflowMaterializationError("completion join requires at least two states")
    attrs = {
        "amoeba.semantic.schema": SCHEMA,
        "amoeba.semantic.graph_id": graph.get("graph_id", ""),
        "amoeba.semantic.task_role": "completion_only",
        "amoeba.semantic.join_kind": kind,
        "amoeba.semantic.source_provenance": "canonical_source_to_joint_rewrite",
    }
    attr_text = ", ".join(f"{key} = {_quote(value)}" for key, value in attrs.items())
    return (f"    {_ssa('join_' + name)} = taskflow.join states({', '.join(states)}) "
            f"base(%D) axis({axis}) region([{', '.join(map(str, lower))}], "
            f"[{', '.join(map(str, upper))}]) {{{attr_text}}} : {type_c}")


def _completion_joins(graph: Mapping[str, Any], terminal_values: Mapping[str, str],
                      task_by_id: Mapping[str, Mapping[str, Any]], type_c: str) -> tuple[list[str], str]:
    by_row: dict[tuple[int, int], list[tuple[list[int], list[int], str]]] = defaultdict(list)
    for task_id, state in terminal_values.items():
        task = task_by_id[task_id]
        by_row[(int(task["m_range"][0]), int(task["m_range"][1]))].append(
            (list(task["m_range"]), list(task["n_range"]), state))
    join_lines: list[str] = []
    row_states: list[tuple[list[int], str]] = []
    for row_key in sorted(by_row):
        entries = sorted(by_row[row_key], key=lambda item: item[1][0])
        if len(entries) == 1:
            row_states.append((entries[0][0], entries[0][2]))
            continue
        m_range = entries[0][0]
        state_values = [entry[2] for entry in entries]
        n_end = entries[-1][1][1]
        name = f"row_{m_range[0]}_{m_range[1]}"
        join_lines.append(_join_line(name, state_values, [m_range[0], 0],
                                      [m_range[1], n_end], 1, graph, type_c))
        row_states.append((m_range, _ssa("join_" + name)))
    if len(row_states) == 1:
        return join_lines, row_states[0][1]
    states = [state for _, state in row_states]
    m_end = row_states[-1][0][1]
    join_lines.append(_join_line("all_output_tiles", states, [0, 0],
                                 [m_end, int(graph["dimensions"]["N"])], 0,
                                 graph, type_c))
    return join_lines, _ssa("join_all_output_tiles")


def materialize_joint_taskflow(identity_or_graph: Mapping[str, Any],
                               graph: Mapping[str, Any] | None = None) -> str:
    """Materialize one validated semantic joint graph as Taskflow MLIR."""

    if graph is None:
        graph = _graph(identity_or_graph)
    else:
        graph = dict(_graph(graph))
        supplied_identity = (identity_or_graph.get("program_identity",
                                                   identity_or_graph)
                             if isinstance(identity_or_graph, Mapping)
                             else None)
        if isinstance(supplied_identity, Mapping):
            graph.setdefault("program_identity", supplied_identity)
            graph.setdefault("source_identity", supplied_identity)
    graph = _graph(graph)
    rewrite = _rewrite_module()
    try:
        rewrite.validate_graph(graph, raise_on_error=True)
    except Exception as error:
        if isinstance(error, TaskflowMaterializationError):
            raise
        raise TaskflowMaterializationError(f"graph is not semantically legal: {error}") from error
    identity = _identity(graph)
    m, n, k = _dimensions(graph)
    type_a, type_b, type_c = _memref(m, k), _memref(k, n), _memref(m, n)
    incoming, outgoing = _edges(graph)
    tasks = _topological(graph, incoming, outgoing)
    task_by_id = {_task_id(task): task for task in graph.get("tasks", [])
                  if isinstance(task, Mapping)}

    # T is an externally visible intermediate only while a plain producer
    # writes it or a plain consumer/reduction reads or writes it.  Once the
    # graph marks the pipeline as fused, sequential producers carry private
    # state and parallel producers carry private partials; allocating a
    # global T in those variants would reintroduce the eliminated buffer.
    global_t_needed = any(
        _task_kind(task) in ("elementwise_consumer", "reduction") or
        (_task_kind(task) == "gemm_producer" and
         task.get("final_C_ownership") == "external_final_C")
        for task in tasks)
    state_tasks = [task for task in tasks
                   if _task_kind(task) == "gemm_producer" and
                   task.get("k_policy") == "sequential" and
                   task.get("final_C_ownership") != "external_final_C"]
    partial_tasks = [task for task in tasks
                     if _task_kind(task) == "gemm_producer" and
                     task.get("k_policy") == "parallel"]
    allocations: dict[str, str] = {}
    if global_t_needed:
        allocations["T"] = "%T"
    for task in state_tasks:
        allocations["state:" + _task_id(task)] = "%state_" + _ident(_task_id(task))
    for task in partial_tasks:
        regions = task.get("regions", {})
        partial = regions.get("partial_output") if isinstance(regions, Mapping) else None
        buffer = _buf(partial) if partial is not None else "partial_T/" + _task_id(task)
        allocations[buffer] = "%partial_" + _ident(_task_id(task))

    function_name = _ident(identity.get("function", "gemm_then_add_8x8_i32"))
    source_id = graph.get("canonical_source_id", identity.get("canonical_pipeline_id", ""))
    lines = [
        "module {",
        f"  func.func @{function_name}(",
        f"      %A: {type_a} {{amoeba.noalias}},",
        f"      %B: {type_b} {{amoeba.noalias}},",
        f"      %C_initial: {type_c} {{amoeba.noalias}},",
        f"      %E: {type_c} {{amoeba.noalias}},",
        f"      %D: {type_c} {{amoeba.noalias}}) -> {type_c} attributes {{",
        f"    amoeba.semantic.schema = {_quote(SCHEMA)},",
        f"    amoeba.semantic.graph_id = {_quote(graph.get('graph_id', ''))},",
        f"    amoeba.semantic.canonical_source_id = {_quote(source_id)},",
        "    amoeba.semantic.arithmetic_semantics = \"i32_modulo_2^32\",",
        "    amoeba.semantic.fusion_layer = \"task_graph_fusion\",",
        "    amoeba.semantic.source_provenance = \"canonical_source_to_joint_rewrite\"",
        "  } {",
    ]
    for buffer, allocation in allocations.items():
        if buffer == "T":
            lines.append(f"    {allocation} = memref.alloc() : {type_c}")
        else:
            lines.append(f"    {allocation} = memref.alloc() : {type_c}")

    values: dict[str, str] = {}
    for task in tasks:
        task_id, kind = _task_id(task), _task_kind(task)
        in_edges = incoming.get(task_id, [])
        out_edges = outgoing.get(task_id, [])
        reads: list[str] = []
        originals: list[str] = []
        input_regions: list[Any] = []
        writes: list[str] = []
        write_originals: list[str] = []
        output_regions: list[Any] = []

        def predecessor(edge_kind: str | None = None) -> str | None:
            for edge in in_edges:
                if edge_kind is None or edge.get("kind") == edge_kind:
                    source = str(edge["source_task"])
                    if source in values:
                        return values[source]
            return None

        if kind == "gemm_producer":
            reads.extend(["%A", "%B"])
            originals.extend(["%A", "%B"])
            if task.get("k_policy") == "sequential":
                state = ("%C_initial" if task.get("initial_C_ownership") == "initial_C_once"
                         else predecessor("state_carry"))
                if state is None:
                    raise TaskflowMaterializationError(f"state carry missing for {task_id}")
                reads.append(state)
                originals.append("%C_initial" if state == "%C_initial" else state)
                output = (allocations.get("state:" + task_id)
                          if task.get("final_C_ownership") != "external_final_C"
                          else allocations.get("T"))
                if output is None:
                    raise TaskflowMaterializationError(f"producer {task_id} has no output storage")
                writes.append(output)
                write_originals.append(output)
            else:
                regions = task.get("regions", {})
                partial = regions.get("partial_output") if isinstance(regions, Mapping) else None
                buffer = _buf(partial) if partial is not None else "partial_T/" + task_id
                output = allocations.get(buffer)
                if output is None:
                    raise TaskflowMaterializationError(f"partial output storage missing for {task_id}")
                writes.append(output)
                write_originals.append(output)
        elif kind == "elementwise_consumer":
            state = predecessor("producer_consumer")
            if state is None:
                state = predecessor()
            if state is None:
                raise TaskflowMaterializationError(f"consumer {task_id} has no producer state")
            reads.extend([state, "%E"])
            originals.extend([allocations.get("T", state), "%E"])
            writes.append("%D")
            write_originals.append("%D")
        elif kind == "fused_producer_consumer":
            reads.extend(["%A", "%B"])
            originals.extend(["%A", "%B"])
            state = ("%C_initial" if task.get("initial_C_ownership") == "initial_C_once"
                     else predecessor("state_carry"))
            if state is None:
                raise TaskflowMaterializationError(f"fused producer {task_id} has no C state")
            reads.extend([state, "%E"])
            originals.extend(["%C_initial" if state == "%C_initial" else state, "%E"])
            writes.append("%D")
            write_originals.append("%D")
        elif kind in ("reduction", "fused_reduction_consumer"):
            partial_edges = [edge for edge in in_edges if edge.get("kind") == "partial_reduction"]
            partial_edges.sort(key=lambda edge: str(edge.get("source_task")))
            if not partial_edges:
                raise TaskflowMaterializationError(f"reduction {task_id} has no partial inputs")
            for edge in partial_edges:
                source = str(edge["source_task"])
                if source not in values:
                    raise TaskflowMaterializationError(f"reduction input {source} is unavailable")
                reads.append(values[source])
                producer = task_by_id[source]
                pregions = producer.get("regions", {})
                pregion = pregions.get("partial_output") if isinstance(pregions, Mapping) else None
                buffer = _buf(pregion) if pregion is not None else "partial_T/" + source
                originals.append(allocations.get(buffer, values[source]))
            reads.append("%C_initial")
            originals.append("%C_initial")
            if kind == "fused_reduction_consumer":
                reads.append("%E")
                originals.append("%E")
                writes.append("%D")
                write_originals.append("%D")
            else:
                output = allocations.get("T")
                if output is None:
                    raise TaskflowMaterializationError("reduction needs global T storage")
                writes.append(output)
                write_originals.append(output)
        else:
            raise TaskflowMaterializationError(f"unsupported task kind {kind!r}")

        input_regions, output_regions = _task_regions_for_operation(graph, task, reads, writes)
        read_types = [type_a if value == "%A" else type_b if value == "%B" else type_c
                      for value in reads]
        all_types = read_types + [type_c] * len(writes)
        result_name = _ssa(task_id)
        lines.append(f"    {result_name} = taskflow.task @{_ident(task_id)}")
        lines.append("        will_reads(" + ", ".join(reads) + " : " +
                     ", ".join(read_types) + ")")
        lines.append("        will_writes(" + ", ".join(writes) + " : " +
                     ", ".join(type_c for _ in writes) + ")")
        lines.append("        [original_read_memrefs(" + ", ".join(originals) + " : " +
                     ", ".join(type_a if value == "%A" else type_b if value == "%B" else type_c
                               for value in originals) + "),")
        lines.append("         original_write_memrefs(" + ", ".join(write_originals) + " : " +
                     ", ".join(type_c for _ in write_originals) + ")]")
        attrs = _task_attrs(graph, task, in_edges, out_edges, input_regions, output_regions)
        lines.append("        {")
        lines.extend("          " + attr + ("," if index + 1 < len(attrs) else "")
                     for index, attr in enumerate(attrs))
        lines.append("        }")
        lines.append("        : (" + ", ".join(all_types) + ") -> (" + type_c + ") {")
        lines.append("      ^bb0(" + ", ".join(
            f"%arg{index}: {typ}" for index, typ in enumerate(all_types)) + "):")
        if kind == "gemm_producer":
            lines.extend(_body_gemm(task, read_count=len(reads), write_arg=len(reads),
                                    parallel=task.get("k_policy") == "parallel",
                                    fused=False, e_arg=None, type_c=type_c))
        elif kind == "fused_producer_consumer":
            lines.extend(_body_gemm(task, read_count=len(reads), write_arg=len(reads),
                                    parallel=False, fused=True, e_arg=3, type_c=type_c))
        elif kind == "elementwise_consumer":
            lines.extend(_body_consumer(task, write_arg=len(reads), type_c=type_c))
        else:
            lines.extend(_body_reduction(
                task, partial_count=len(reads) - (2 if kind == "fused_reduction_consumer" else 1),
                write_arg=len(reads), fused=kind == "fused_reduction_consumer",
                e_arg=len(reads) - 1 if kind == "fused_reduction_consumer" else None,
                type_c=type_c))
        lines.append("    }")
        values[task_id] = result_name

    terminal_ids = [task_id for task_id in graph.get("terminal_task_ids", [])
                    if task_id in values]
    if not terminal_ids:
        for task in tasks:
            task_id = _task_id(task)
            if not any(edge.get("source_task") == task_id and
                       edge.get("target_task") in task_by_id and
                       _task_kind(task_by_id[str(edge["target_task"])]) != "completion_join"
                       for edge in outgoing.get(task_id, [])):
                terminal_ids.append(task_id)
    terminal_values = {task_id: values[task_id] for task_id in terminal_ids}
    join_lines, final_value = _completion_joins(graph, terminal_values, task_by_id, type_c)
    lines.extend(join_lines)
    lines.append(f"    return {final_value} : {type_c}")
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def materialize_taskflow(identity_or_graph: Mapping[str, Any],
                         graph: Mapping[str, Any] | None = None) -> str:
    """Compatibility alias used by the semantic candidate drivers."""

    return materialize_joint_taskflow(identity_or_graph, graph)


__all__ = ["SCHEMA", "TaskflowMaterializationError",
           "materialize_joint_taskflow", "materialize_taskflow"]
