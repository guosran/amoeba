#!/usr/bin/env python3
"""Fail-closed comparison of semantic joint graphs and C++ Taskflow facts.

The graph rewrite enumerator owns algorithmic edge roles.  The C++ extractor
independently proves that each annotated edge exists in Taskflow, including
its storage scope and half-open transfer region.  This module joins those two
views without using backend, cost, or schedule information.
"""

from __future__ import annotations

from collections import Counter
import copy
from itertools import permutations, product
from typing import Any, Mapping


class GraphFactsMismatch(ValueError):
    pass


def _region(edge: Mapping[str, Any], *, cpp: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if cpp:
        lower, upper = edge.get("region_lower"), edge.get("region_upper")
    else:
        value = edge.get("region", edge.get("transfer_region"))
        if not isinstance(value, Mapping):
            raise GraphFactsMismatch("semantic edge has no transfer region")
        lower, upper = value.get("lower"), value.get("upper")
    if (not isinstance(lower, list) or not isinstance(upper, list) or
            len(lower) != len(upper) or not lower or
            any(type(item) is not int for item in lower + upper) or
            any(lo >= hi for lo, hi in zip(lower, upper))):
        raise GraphFactsMismatch("edge has an invalid half-open transfer region")
    return tuple(lower), tuple(upper)


def compare_graph_facts(graph: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """Compare exact task IDs and data-edge (kind, role, scope, region) tuples.

    Completion joins are Taskflow join operations, not numeric tasks or data
    edges.  They are checked separately through the C++ completion_joins list.
    """
    tasks = graph.get("tasks")
    edges = graph.get("edges", graph.get("typed_edges"))
    if not isinstance(tasks, list) or not isinstance(edges, list):
        raise GraphFactsMismatch("semantic graph has no tasks or typed edges")
    expected_tasks = {
        task["task_id"] for task in tasks
        if task.get("task_role", task.get("role", task.get("task_kind")))
        not in ("completion_join", "completion_only")
    }
    actual_tasks = {task["task_name"] for task in facts.get("tasks", [])}
    if len(expected_tasks) != len([task for task in tasks if
           task.get("task_role", task.get("role", task.get("task_kind")))
           not in ("completion_join", "completion_only")]):
        raise GraphFactsMismatch("semantic task IDs are duplicated")

    expected = Counter()
    for edge in edges:
        role = edge.get("kind", edge.get("semantic_role"))
        if role in ("completion", "completion_only"):
            continue
        if role is None or edge.get("scope") not in ("tile_local", "tensor_wide"):
            raise GraphFactsMismatch("semantic edge lacks role or scope")
        if edge.get("source_task") not in expected_tasks or edge.get("target_task") not in expected_tasks:
            raise GraphFactsMismatch("semantic data edge names a missing task")
        expected[(edge["source_task"], edge["target_task"], "raw", role,
                  edge["scope"], *_region(edge, cpp=False))] += 1
    actual = Counter()
    for edge in facts.get("dependencies", []):
        role = edge.get("semantic_role")
        scope = edge.get("semantic_scope")
        if role is None or scope is None or scope != edge.get("scope"):
            raise GraphFactsMismatch("C++ data edge lacks verified semantic annotation")
        actual[(edge["source_task"], edge["target_task"], edge["kind"],
                role, scope, *_region(edge, cpp=True))] += 1
    missing = expected - actual
    unexpected = actual - expected
    completion_expected = graph.get("completion_join_materialized")
    completion_count = len(facts.get("completion_joins", []))
    completion_matches = (completion_expected is None or
                          bool(completion_expected) == bool(completion_count))
    pass_ = (expected_tasks == actual_tasks and not missing and
             not unexpected and completion_matches)
    return {
        "pass": pass_,
        "expected_task_ids": sorted(expected_tasks),
        "actual_task_ids": sorted(actual_tasks),
        "expected_data_edge_count": sum(expected.values()),
        "actual_data_edge_count": sum(actual.values()),
        "missing_edges": [[*key[:5], list(key[5]), list(key[6]), count]
                          for key, count in sorted(missing.items())],
        "unexpected_edges": [[*key[:5], list(key[5]), list(key[6]), count]
                             for key, count in sorted(unexpected.items())],
        "completion_join_count": completion_count,
        "completion_join_matches": completion_matches,
    }


def compare_graph_facts_by_semantic_signature(
        graph: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """Prove a unique role/K/output-region task bijection before edge matching.

    Standalone MLIR passes assign operation names while the enumerator assigns
    graph IDs.  Names may differ with rewrite order.  A task mapping is accepted
    only if its role, K interval and proven output rectangle identify exactly
    one task on both sides; exact typed edge tuples are checked afterward.
    """
    def interval(value: Any) -> tuple[int, ...]:
        if not isinstance(value, list) or any(type(x) is not int for x in value):
            raise GraphFactsMismatch("task has no proven K interval")
        return tuple(value)

    def output_region(task: Mapping[str, Any], *, cpp: bool) -> tuple[Any, ...]:
        if cpp:
            regions = task.get("output_regions")
            if not isinstance(regions, list) or len(regions) != 1:
                raise GraphFactsMismatch("C++ task lacks one proven output region")
            region = regions[0]
        else:
            regions = task.get("regions", {})
            writes = task.get("original_write_memrefs", [])
            name = ("D_output" if writes == ["D"] else
                    "T_output" if writes == ["T"] else
                    "partial_output" if len(writes) == 1 and
                    str(writes[0]).startswith("partial_T/") else None)
            if writes == ["D"] and name not in regions and "T_output" in regions:
                # A fused reduction publishes D directly while the graph's
                # inherited reduction rectangle is still named T_output.
                name = "T_output"
            if name is None or name not in regions:
                raise GraphFactsMismatch("semantic task lacks one output region")
            region = regions[name]
        if not isinstance(region, Mapping):
            raise GraphFactsMismatch("invalid task output region")
        return interval(region.get("lower")), interval(region.get("upper"))

    def signature(task: Mapping[str, Any], *, cpp: bool) -> tuple[Any, ...]:
        role = task.get("semantic_role") if cpp else task.get("task_role")
        k_range = task.get("semantic_k_range") if cpp else task.get("K_range")
        if not isinstance(role, str) or not role:
            raise GraphFactsMismatch("task has no semantic role")
        # The elementwise consumer has no K iteration.  A source annotation
        # may carry the producer's full K interval only as provenance.
        k_signature = () if role == "consumer" else interval(k_range)
        return role, k_signature, *output_region(task, cpp=cpp)

    expected_tasks = [task for task in graph.get("tasks", []) if
                      task.get("task_role") not in
                      ("completion_join", "completion_only")]
    actual_tasks = facts.get("tasks", [])
    if len(expected_tasks) != len(actual_tasks):
        raise GraphFactsMismatch("semantic and C++ task counts differ")
    expected_base = {task["task_id"]: signature(task, cpp=False)
                     for task in expected_tasks}
    actual_base = {task["task_name"]: signature(task, cpp=True)
                   for task in actual_tasks}

    def neighborhoods(*, cpp: bool) -> dict[str, tuple[Any, ...]]:
        bases = actual_base if cpp else expected_base
        adjacency: dict[str, list[tuple[Any, ...]]] = {
            name: [] for name in bases}
        edges = (facts.get("dependencies", []) if cpp else
                 graph.get("edges", graph.get("typed_edges", [])))
        for edge in edges:
            role = (edge.get("semantic_role") if cpp else
                    edge.get("kind", edge.get("semantic_role")))
            if role in ("completion", "completion_only"):
                continue
            source, target = edge["source_task"], edge["target_task"]
            if source not in bases or target not in bases:
                raise GraphFactsMismatch("typed edge names a missing task")
            scope = edge.get("semantic_scope") if cpp else edge.get("scope")
            label = (role, scope, *_region(edge, cpp=cpp))
            adjacency[source].append(("out", label, bases[target]))
            adjacency[target].append(("in", label, bases[source]))
        return {name: tuple(sorted(edges, key=repr))
                for name, edges in adjacency.items()}

    expected_neighborhoods = neighborhoods(cpp=False)
    actual_neighborhoods = neighborhoods(cpp=True)
    actual_by_signature: dict[tuple[Any, ...], list[str]] = {}
    for name, base in actual_base.items():
        key = base, actual_neighborhoods[name]
        actual_by_signature.setdefault(key, []).append(name)
    expected_by_signature: dict[tuple[Any, ...], list[str]] = {}
    for name, base in expected_base.items():
        key = base, expected_neighborhoods[name]
        expected_by_signature.setdefault(key, []).append(name)
    if (set(expected_by_signature) != set(actual_by_signature) or
            any(len(expected_by_signature[key]) != len(actual_by_signature[key])
                for key in expected_by_signature)):
        raise GraphFactsMismatch("no C++ task bijection for semantic signatures")

    # Internal tree reductions and the final reduction can have the same
    # role, K interval, and output rectangle.  Their typed incoming/outgoing
    # edges disambiguate them; a remaining graph automorphism is rejected.
    groups = sorted(expected_by_signature, key=repr)
    options = [list(permutations(actual_by_signature[key])) for key in groups]
    combinations = 1
    for choices in options:
        combinations *= len(choices)
    if combinations > 4096:
        raise GraphFactsMismatch("task bijection search exceeds bound")
    matching: list[tuple[dict[str, Any], dict[str, str]]] = []
    for choices in product(*options):
        rename = {source: target for key, permutation in zip(groups, choices)
                  for source, target in zip(expected_by_signature[key], permutation)}
        converted = copy.deepcopy(graph)
        for task in converted["tasks"]:
            if task["task_id"] in rename:
                task["task_id"] = rename[task["task_id"]]
        for edge in converted.get("edges", converted.get("typed_edges", [])):
            if edge.get("source_task") in rename:
                edge["source_task"] = rename[edge["source_task"]]
            if edge.get("target_task") in rename:
                edge["target_task"] = rename[edge["target_task"]]
        result = compare_graph_facts(converted, facts)
        if result["pass"]:
            matching.append((result, rename))
            if len(matching) > 1:
                raise GraphFactsMismatch("typed graph task bijection is ambiguous")
    if not matching:
        raise GraphFactsMismatch("no C++ task bijection matches typed edges")
    result, rename = matching[0]
    result["task_id_bijection"] = rename
    result["bijection_basis"] = (
        "semantic_role_k_interval_output_region_and_typed_edges")
    return result


__all__ = ["GraphFactsMismatch", "compare_graph_facts",
           "compare_graph_facts_by_semantic_signature"]
