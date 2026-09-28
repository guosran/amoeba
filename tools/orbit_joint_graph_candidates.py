#!/usr/bin/env python3
"""Reference-model candidate publisher for cross-checking the MLIR pass.

The production ORBIT entry is ``orbit_joint_mlir_pass_entry.py`` and invokes
the C++ MLIR pass.  This reference model is retained as an independent
numeric/structural regression oracle for the small canonical fixture.  It
never calls mapping, cost, replay, placement, replication, or a hardware
backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from orbit_joint_pipeline import canonical_pipeline_identity
from orbit_joint_rewrite_space import (canonical_graph_id,
                                       enumerate_rewrite_dag, root_graph,
                                       validate_graph)


SCHEMA = "orbit-joint-semantic-graph-candidates-v1"
DEFAULT_OPT = Path(
    "/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/mlir-amoeba-opt")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True,
                               indent=2) + "\n", encoding="utf-8")


def _factors(text: str) -> list[int]:
    try:
        factors = sorted(set(int(item) for item in text.split(",")))
    except ValueError as error:
        raise ValueError("tile factors must be positive integers") from error
    if not factors or factors[0] <= 0:
        raise ValueError("tile factors must be positive integers")
    return factors


def _actions(dimensions: Mapping[str, int], factors: Sequence[int]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for name, axis in (("TileOutputM", "M"), ("TileOutputN", "N"),
                       ("TileReductionKSequential", "K"),
                       ("TileReductionKParallelLinear", "K"),
                       ("TileReductionKParallelTree", "K")):
        for factor in factors:
            # The untiled M/N/K state is the DAG root.  An equal factor is a
            # no-op, not another graph.  The prior standalone GEMM evidence
            # keeps its 192 candidates independently of this pipeline DAG.
            if factor < dimensions[axis]:
                actions.append({"kind": name, "tile_size": factor})
    actions.extend({"kind": name} for name in (
        "FuseProducerConsumer", "FuseTileLocalPair",
        "MaterializeNumericReduction", "FuseReductionConsumer",
        "MaterializeCompletionJoin"))
    return actions


def _record(graph: Mapping[str, Any], index: int, source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "record_type": "candidate",
        "index": index,
        "graph_variant_id": canonical_graph_id(graph),
        "canonical_pipeline_id": source["canonical_pipeline_id"],
        "source_module_sha256": source["source_module_sha256"],
        "candidate_kind": "semantic_graph_candidate",
        "semantic_legal": True,
        "backend_status": "unknown",
        "graph_facts_sha256": _sha(_json({
            "tasks": graph["tasks"], "typed_edges": graph["edges"]})),
        "task_count": len(graph["tasks"]),
        "typed_edge_count": len(graph["edges"]),
        "k_policy": graph["k_policy"],
        "reduction_topology": graph["reduction_topology"],
        "fusion": graph["fusion"],
        "intermediate": graph["intermediate"],
        "m_tiles": graph["m_tiles"],
        "n_tiles": graph["n_tiles"],
        "k_blocks": graph["k_blocks"],
        "fusion_layers": {
            "task_graph_fusion": bool(graph["mode_tags"]["fused"]),
            "kernel_dfg_fusion": False,
            "fu_fusion": False,
            "streaming_memory_fusion": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.input.resolve()
    if not source_path.is_file() or not args.amoeba_opt.is_file():
        raise ValueError("canonical source or mlir-amoeba-opt does not exist")
    source_bytes = source_path.read_bytes()
    verified = subprocess.run([
        str(args.amoeba_opt), str(source_path), "--verify-each", "-o", "/dev/null"],
        text=True, capture_output=True)
    if verified.returncode:
        raise ValueError("canonical Taskflow verifier rejected source: " +
                         verified.stderr[-1200:])
    identity = canonical_pipeline_identity(source_path)
    if identity["source_module_sha256"] != _sha(source_bytes):
        raise ValueError("canonical source hash changed during recognition")
    factors = _factors(args.tile_factors)
    root = root_graph(identity)
    dag = enumerate_rewrite_dag(
        root, max_depth=args.max_depth, max_actions=args.max_actions,
        actions=_actions(root["dimensions"], factors))
    if dag["statistics"]["budget_exhausted"]:
        raise ValueError("action budget exhausted; no partial candidate manifest published")
    # Incomplete parallel construction states are DAG vertices, never
    # publishable candidates.  A numeric reduction action must complete them.
    candidates = [graph for graph in dag["graphs"] if validate_graph(graph)]
    if not candidates:
        raise ValueError("rewrite DAG produced no complete semantic graph")
    ids = [canonical_graph_id(graph) for graph in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("rewrite DAG retained duplicate graph identities")
    by_id = dict(zip(ids, candidates))
    selected_id = args.select_variant_id or canonical_graph_id(root)
    if selected_id not in by_id:
        raise ValueError("selected semantic graph ID is absent from complete DAG")
    from orbit_joint_taskflow import materialize_joint_taskflow
    selected_ir = materialize_joint_taskflow(identity, by_id[selected_id])
    with tempfile.TemporaryDirectory(prefix="orbit-joint-candidates-") as dirname:
        temporary = Path(dirname)
        selected_path = temporary / "selected_taskflow.mlir"
        selected_path.write_text(selected_ir, encoding="utf-8")
        result = subprocess.run([
            str(args.amoeba_opt), str(selected_path), "--verify-each", "-o",
            "/dev/null"], text=True, capture_output=True)
        if result.returncode:
            raise ValueError("selected Taskflow materialization is invalid: " +
                             result.stderr[-1200:])
        records = [_record(graph, index, identity)
                   for index, graph in enumerate(candidates)]
        lines = [{"schema": SCHEMA, "record_type": "header", "complete": True,
                  "canonical_pipeline_id": identity["canonical_pipeline_id"],
                  "source_module_sha256": identity["source_module_sha256"],
                  "owner": "tools/orbit_joint_rewrite_space.py",
                  "candidate_kind": "semantic_graph_candidate",
                  "backend_status": "unknown"}, *records,
                 {"schema": SCHEMA, "record_type": "footer", "complete": True,
                  "candidate_count": len(candidates),
                  "selected_graph_variant_id": selected_id}]
        manifest = temporary / "graph_candidate_manifest.jsonl"
        manifest.write_text("".join(_json(line) + "\n" for line in lines),
                            encoding="utf-8")
        _write(temporary / "canonical_pipeline_identity.json", identity)
        _write(temporary / "rewrite_dag_statistics.json", {
            "schema": dag["schema"], "statistics": dag["statistics"],
            "complete_candidate_count": len(candidates),
            "incomplete_construction_state_count":
                len(dag["graphs"]) - len(candidates),
            "accepted_transitions": dag["accepted_transitions"],
            "rejected_transitions": dag["rejected_transitions"],
            "action_catalog": dag["action_catalog"],
        })
        _write(temporary / "joint_variant_space.json", {
            "schema": SCHEMA, "canonical_pipeline_id":
                identity["canonical_pipeline_id"],
            "candidate_count": len(candidates), "candidates": candidates,
        })
        _write(temporary / "source_to_joint_variant_provenance.json", {
            "schema": "orbit-source-to-joint-variant-v1",
            "source_path": str(source_path),
            "source_module_sha256": identity["source_module_sha256"],
            "canonical_pipeline_id": identity["canonical_pipeline_id"],
            "graph_variant_ids": ids,
            "graph_candidate_manifest_sha256": _sha(manifest.read_bytes()),
        })
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        for item in temporary.iterdir():
            os.replace(item, output / item.name)
    return {"schema": SCHEMA, "canonical_pipeline_id":
            identity["canonical_pipeline_id"],
            "candidate_count": len(candidates),
            "rewrite_dag_unique_states": dag["statistics"]["unique_graph_states"],
            "selected_graph_variant_id": selected_id,
            "selected_taskflow_verifier_pass": True,
            "backend_status": "unknown"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--tile-factors", default="4")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--select-variant-id", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parse_args(argv)), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"joint semantic graph candidates failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
