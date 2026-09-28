#!/usr/bin/env python3
"""Focused semantic tests for :mod:`orbit_joint_rewrite_space`."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import orbit_joint_rewrite_space as space


def apply(graph, *actions):
    current = graph
    for action in actions:
        result = space.apply_action(current, action)
        assert result.accepted, (action, result.reason_code, result.diagnostic)
        current = result.graph
    return current


def action(kind, **parameters):
    return {"kind": kind, **parameters}


def inputs(seed=7, dimensions=(8, 8, 8)):
    m, n, k = dimensions
    rng = random.Random(seed)
    return {
        "A": [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(m * k)],
        "B": [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(k * n)],
        "C_initial": [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(m * n)],
        "E": [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(m * n)],
    }


def assert_valid(graph):
    audit = space.validate_graph(graph, return_audit=True)
    assert audit["valid"], audit
    assert graph["graph_id"] == space.canonical_graph_id(graph)


def test_root_schema_and_identity_are_source_bound():
    graph = space.root_graph()
    assert_valid(graph)
    assert graph["k_policy"] == "sequential"
    assert graph["reduction_topology"] == "none"
    assert [task["task_kind"] for task in graph["tasks"]] == [
        "gemm_producer", "elementwise_consumer"]
    assert graph["edges"][0]["kind"] == "producer_consumer"
    assert graph["edges"][0]["scope"] == "tile_local"
    assert graph["edges"][0]["region"]["half_open"] is True
    assert graph["classification"] == "identity"


def test_identity_and_fusion_only():
    graph = apply(space.root_graph(), action("FuseProducerConsumer"))
    assert_valid(graph)
    assert graph["classification"] == "fusion_only"
    assert len(graph["tasks"]) == 1
    task = graph["tasks"][0]
    assert task["task_kind"] == "fused_producer_consumer"
    assert task["writes"] == ["D"]
    assert "E" in task["reads"]
    assert graph["intermediate"]["state"] == "eliminated"


def test_tile_local_fusion_and_equivalent_order_dedup():
    tile_then_fuse = apply(
        space.root_graph(),
        action("TileOutputM", tile_size=4),
        action("TileOutputN", tile_size=4),
        action("FuseTileLocalPair"),
    )
    fuse_then_tile = apply(
        space.root_graph(),
        action("FuseProducerConsumer"),
        action("TileOutputM", tile_size=4),
        action("TileOutputN", tile_size=4),
    )
    assert_valid(tile_then_fuse)
    assert_valid(fuse_then_tile)
    assert space.canonical_graph_id(tile_then_fuse) == space.canonical_graph_id(fuse_then_tile)
    assert space.classify_graph(tile_then_fuse)["modes"] == ["tile_then_fuse"]
    assert space.classify_graph(fuse_then_tile)["modes"] == ["fuse_then_tile"]
    assert len(tile_then_fuse["tasks"]) == 4
    assert all(task["task_kind"] == "fused_producer_consumer"
               for task in tile_then_fuse["tasks"])


def test_sequential_last_k_fusion_preserves_state_chain():
    graph = apply(
        space.root_graph(),
        action("TileReductionKSequential", tile_size=4),
        action("FuseProducerConsumer"),
    )
    assert_valid(graph)
    assert "sequential_last_fusion" in space.classify_graph(graph)["modes"]
    producers = [task for task in graph["tasks"]
                 if task["task_kind"] in ("gemm_producer", "fused_producer_consumer")]
    assert [task["initial_C_ownership"] for task in producers] == [
        "initial_C_once", "state_carry"]
    assert producers[-1]["task_kind"] == "fused_producer_consumer"
    assert not any(task["task_kind"] == "elementwise_consumer"
                   for task in graph["tasks"])
    assert [edge["kind"] for edge in graph["edges"]] == ["state_carry"]
    assert all(edge["scope"] == "tile_local" for edge in graph["edges"])


def test_early_sequential_k_fusion_is_rejected():
    graph = apply(space.root_graph(), action("TileReductionKSequential", tile_size=4))
    # Targeting an early block cannot make the consumer complete.  The action
    # checker accepts only the implicit final block when a K chain exists.
    result = space.apply_action(graph, action(
        "FuseProducerConsumer", target_task_ids=["producer_tile_m0_n0_k0_0_4"]))
    assert not result.accepted
    assert result.reason_code == "REGION_MISMATCH"


def test_parallel_linear_reduction_then_fusion():
    graph = apply(
        space.root_graph(),
        action("TileReductionKParallelLinear", tile_size=4),
    )
    assert not space.validate_graph(graph)
    assert space.validate_graph(graph, allow_incomplete=True)
    graph = apply(graph, action("MaterializeNumericReduction"),
                  action("FuseReductionConsumer"))
    assert_valid(graph)
    assert graph["reduction_topology"] == "linear"
    assert graph["intermediate"]["private_partial"] is True
    assert len([task for task in graph["tasks"]
                if task["task_kind"] == "gemm_producer"]) == 2
    assert graph["tasks"][-1]["task_kind"] == "fused_reduction_consumer"
    assert all(edge["kind"] == "partial_reduction" for edge in graph["edges"])
    assert "parallel_linear_reduction_fusion" in space.classify_graph(graph)["modes"]


def test_parallel_tree_topology_is_preserved_in_identity():
    linear = apply(
        space.root_graph(),
        action("TileReductionKParallelLinear", tile_size=2),
        action("MaterializeNumericReduction"),
        action("FuseReductionConsumer"),
    )
    tree = apply(
        space.root_graph(),
        action("TileReductionKParallelTree", tile_size=2),
        action("MaterializeNumericReduction"),
        action("FuseReductionConsumer"),
    )
    assert_valid(linear)
    assert_valid(tree)
    assert linear["reduction_topology"] == "linear"
    assert tree["reduction_topology"] == "balanced_tree"
    assert space.canonical_graph_id(linear) != space.canonical_graph_id(tree)
    assert "parallel_tree_reduction_fusion" in space.classify_graph(tree)["modes"]


def test_parallel_direct_partial_consumer_fusion_is_rejected():
    graph = apply(space.root_graph(), action("TileReductionKParallelLinear", tile_size=4))
    result = space.apply_action(graph, action("FuseProducerConsumer"))
    assert not result.accepted
    assert result.reason_code == "REDUCTION_REQUIRED"


def test_completion_join_is_separate_from_numeric_reduction():
    graph = apply(
        space.root_graph(),
        action("TileReductionKParallelLinear", tile_size=4),
        action("MaterializeNumericReduction"),
        action("MaterializeCompletionJoin"),
    )
    assert_valid(graph)
    joins = [task for task in graph["tasks"] if task["task_kind"] == "completion_join"]
    reductions = [task for task in graph["tasks"] if task["task_kind"] == "reduction"]
    assert len(joins) == 1
    assert len(reductions) == 1
    assert joins[0]["completion_only"] is True
    assert joins[0]["numeric"] is False
    assert any(edge["kind"] == "completion" for edge in graph["edges"])
    assert not any(edge["kind"] == "completion" and
                   edge.get("region", {}).get("buffer") == "partial_T"
                   for edge in graph["edges"])


@pytest.mark.parametrize("override,code", [
    ({"intermediate_external_users": ["other"]}, "EXTERNAL_USER_PRESENT"),
    ({"alias_free": False}, "ALIAS_PROOF_MISSING"),
    ({"producer_side_effect_free": False}, "SIDE_EFFECT_UNPROVEN"),
    ({"consumer_unknown_access": True}, "UNKNOWN_ACCESS"),
    ({"producer_consumer_region": "mismatch"}, "REGION_MISMATCH"),
    ({"consumer_halo": True}, "HALO_UNSUPPORTED"),
    ({"consumer_broadcast": True}, "BROADCAST_UNSUPPORTED"),
    ({"consumer_is_reduction": True}, "REDUCTION_CONSUMER_UNSUPPORTED"),
])
def test_fusion_legality_negative_contracts(override, code):
    graph = space.root_graph(contract_overrides=override)
    audit = space.validate_graph(graph, return_audit=True)
    assert not audit["valid"]
    assert audit["reason_code"] == code
    result = space.apply_action(graph, action("FuseProducerConsumer"))
    assert not result.accepted
    assert result.reason_code == code if code != "EXTERNAL_USER_PRESENT" else result.reason_code == code


def test_unknown_action_is_atomic():
    graph = space.root_graph()
    before = copy.deepcopy(graph)
    result = space.apply_action(graph, {"kind": "UnknownRewrite"})
    assert not result.accepted
    assert result.reason_code == "ACTION_UNKNOWN"
    assert graph == before


def test_action_history_does_not_affect_graph_id_or_runtime_inputs():
    graph = apply(space.root_graph(), action("FuseProducerConsumer"))
    same = copy.deepcopy(graph)
    same["action_history"].append({"kind": "runtime_only", "seed": 991})
    assert space.canonical_graph_id(graph) == space.canonical_graph_id(same)
    values = inputs(5)
    first = space.interpret_graph(graph, values)
    second = space.interpret_graph(same, values)
    assert first["D"] == second["D"]


def test_graph_id_changes_for_regions_fusion_and_topology():
    root = space.root_graph()
    fused = apply(root, action("FuseProducerConsumer"))
    tiled = apply(root, action("TileOutputM", tile_size=4))
    tree = apply(root, action("TileReductionKParallelTree", tile_size=4),
                 action("MaterializeNumericReduction"))
    assert len({space.canonical_graph_id(item) for item in (root, fused, tiled, tree)}) == 4


def test_dag_memoizes_duplicate_states_and_reports_rejections():
    dag = space.enumerate_rewrite_dag(max_depth=5, max_actions=1000,
                                     tile_m=4, tile_n=4, tile_k=4)
    stats = dag["statistics"]
    assert dag["root_graph_id"] in dag["graph_ids"]
    assert stats["unique_candidate_graphs"] == len(dag["graphs"])
    assert stats["construction_states"] == len(dag["states"])
    assert stats["accepted_transitions"] > 0
    assert stats["rejected_transitions"] > 0
    assert stats["duplicate_states_removed"] > 0
    modes = {mode for record in dag["modes"].values() for mode in record["modes"]}
    assert {"identity", "tiling_only", "fusion_only"}.issubset(modes)


def test_dag_is_deterministic_under_repeated_runs():
    first = space.enumerate_rewrite_dag(max_depth=4, max_actions=300,
                                        tile_m=4, tile_n=4, tile_k=4)
    second = space.enumerate_rewrite_dag(max_depth=4, max_actions=300,
                                         tile_m=4, tile_n=4, tile_k=4)
    assert first["graph_ids"] == second["graph_ids"]
    assert first["accepted_transitions"] == second["accepted_transitions"]
    assert first["rejected_transitions"] == second["rejected_transitions"]


@pytest.mark.parametrize("actions", [
    [],
    [action("FuseProducerConsumer")],
    [action("TileOutputM", tile_size=4), action("TileOutputN", tile_size=4)],
    [action("TileOutputM", tile_size=4), action("TileOutputN", tile_size=4),
     action("FuseTileLocalPair")],
    [action("FuseProducerConsumer"), action("TileOutputM", tile_size=4),
     action("TileOutputN", tile_size=4)],
    [action("TileReductionKSequential", tile_size=4), action("FuseProducerConsumer")],
    [action("TileReductionKParallelLinear", tile_size=4),
     action("MaterializeNumericReduction"), action("FuseReductionConsumer")],
    [action("TileReductionKParallelTree", tile_size=4),
     action("MaterializeNumericReduction"), action("FuseReductionConsumer")],
    [action("TileOutputM", tile_size=4), action("TileOutputN", tile_size=4),
     action("TileReductionKSequential", tile_size=4), action("FuseTileLocalPair")],
])
def test_numeric_equivalence_and_intermediate_debug(actions):
    graph = apply(space.root_graph(), *actions)
    values = inputs(11)
    actual = space.interpret_graph(graph, values)
    # A direct scalar reference in the same arithmetic domain.
    m = n = k = 8
    expected_t = list(values["C_initial"])
    for row in range(m):
        for column in range(n):
            index = row * n + column
            for kk in range(k):
                expected_t[index] = (
                    expected_t[index] +
                    values["A"][row * k + kk] * values["B"][kk * n + column]
                ) & 0xffffffff
    expected_d = [(left + right) & 0xffffffff
                  for left, right in zip(expected_t, values["E"])]
    assert actual["T"] == expected_t
    assert actual["D"] == expected_d
    assert actual["debug"]["intermediate"]["identity"] == "T"
    if graph["k_policy"] == "parallel":
        assert actual["partials"]
        assert actual["reduction_outputs"]
    else:
        assert actual["state_carry"]


def test_numeric_overflow_wraps_modulo_two_to_thirty_two():
    values = {
        "A": [0x7fffffff] * 64,
        "B": [2] * 64,
        "C_initial": [0xffffffff] * 64,
        "E": [0x80000000] * 64,
    }
    graph = apply(space.root_graph(),
                  action("TileReductionKParallelTree", tile_size=2),
                  action("MaterializeNumericReduction"),
                  action("FuseReductionConsumer"))
    output = space.interpret_graph(graph, values)
    assert all(0 <= value < (1 << 32) for value in output["T"])
    assert all(0 <= value < (1 << 32) for value in output["D"])
    assert output["D"] == space.interpret_graph(
        space.root_graph(), values)["D"]


def test_completion_join_and_numeric_reduction_have_distinct_ids():
    reduced = apply(space.root_graph(), action("TileReductionKParallelLinear", tile_size=4),
                    action("MaterializeNumericReduction"))
    joined = apply(reduced, action("MaterializeCompletionJoin"))
    fused = apply(reduced, action("FuseReductionConsumer"))
    assert space.canonical_graph_id(reduced) != space.canonical_graph_id(joined)
    assert space.canonical_graph_id(reduced) != space.canonical_graph_id(fused)
