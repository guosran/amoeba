#!/usr/bin/env python3
"""Replay representative C++ joint rewrite witnesses through standalone passes.

This is a test driver.  The MLIR passes own enumeration, legality and graph
rewrites; this script only selects witnesses and records verifier/facts results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from orbit_joint_facts_compare import (GraphFactsMismatch,
                                       compare_graph_facts_by_semantic_signature)


def _select(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def find(test: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
        return next(candidate for candidate in candidates if test(candidate))

    def tiled(candidate: Mapping[str, Any], *axes: str) -> bool:
        sizes = candidate["tile_sizes"]
        return all(sizes[f"B{axis}"] < 8 for axis in axes)

    def untiled(candidate: Mapping[str, Any], *axes: str) -> bool:
        sizes = candidate["tile_sizes"]
        return all(sizes[f"B{axis}"] == 8 for axis in axes)

    checks: dict[str, Callable[[Mapping[str, Any]], bool]] = {
        "identity": lambda c: untiled(c, "M", "N", "K") and
            c["fusion"]["producer_consumer"] == "none",
        "fusion_only": lambda c: untiled(c, "M", "N", "K") and
            c["fusion"]["producer_consumer"] == "whole",
        "mn_tiling_only": lambda c: tiled(c, "M", "N") and untiled(c, "K")
            and c["fusion"]["producer_consumer"] == "none"
            and c["completion_join_materialized"],
        "tile_to_fuse": lambda c: tiled(c, "M", "N") and untiled(c, "K")
            and c["fusion"]["producer_consumer"] == "tile_local"
            and c["completion_join_materialized"],
        "fuse_to_tile": lambda c: tiled(c, "M", "N") and untiled(c, "K")
            and c["fusion"]["producer_consumer"] == "tile_local"
            and c["completion_join_materialized"],
        "sequential_k_last_fusion": lambda c: tiled(c, "K") and
            c["k_policy"] == "sequential" and c["fusion"]["sequential_last"],
        "parallel_linear_fusion": lambda c: tiled(c, "K") and
            c["k_policy"] == "parallel" and c["reduction_topology"] == "linear"
            and c["fusion"]["reduction_consumer"],
        "parallel_tree_fusion": lambda c: tiled(c, "K") and
            c["k_policy"] == "parallel" and
            c["reduction_topology"] == "balanced_tree" and
            c["fusion"]["reduction_consumer"],
        "mnk_fusion": lambda c: tiled(c, "M", "N", "K") and
            c["k_policy"] == "parallel" and
            c["reduction_topology"] == "linear" and
            c["fusion"]["reduction_consumer"] and
            c["completion_join_materialized"],
    }
    return {label: find(predicate) for label, predicate in checks.items()}


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True)
    return {"command": command, "returncode": result.returncode,
            "stderr": result.stderr[-3000:]}


def _pass_flag(step: str) -> str:
    if "{" not in step:
        return "--" + step
    name, options = step.split("{", 1)
    if not options.endswith("}"):
        raise ValueError(f"invalid MLIR pass witness step: {step}")
    return "--" + name + "=" + options[:-1]


def run(source: Path, manifest_path: Path, optimizer: Path,
        output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("C++ rewrite manifest is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select(manifest["candidates"])
    records: dict[str, Any] = {}
    for label, candidate in selected.items():
        witness = candidate["witness"]
        pipeline = witness["pass_pipeline"]
        if label in ("tile_to_fuse", "fuse_to_tile"):
            sizes = candidate["tile_sizes"]
            m = f"tile-output-m{{tile-size={sizes['BM']}}}"
            n = f"tile-output-n{{tile-size={sizes['BN']}}}"
            pipeline = ([m, n, "fuse-tile-local-pair"] if
                        label == "tile_to_fuse" else
                        ["fuse-producer-consumer", m, n])
        elif label == "mnk_fusion":
            sizes = candidate["tile_sizes"]
            pipeline = [
                f"tile-reduction-k-parallel-linear{{tile-size={sizes['BK']}}}",
                "materialize-numeric-reduction", "fuse-reduction-consumer",
                f"tile-output-m{{tile-size={sizes['BM']}}}",
                f"tile-output-n{{tile-size={sizes['BN']}}}",
            ]
        elif label == "sequential_k_last_fusion":
            pipeline = [
                f"tile-reduction-k-sequential{{tile-size={candidate['tile_sizes']['BK']}}}",
                "fuse-reduction-consumer",
            ]
        elif label in ("parallel_linear_fusion", "parallel_tree_fusion"):
            topology = "linear" if label == "parallel_linear_fusion" else "tree"
            pipeline = [
                f"tile-reduction-k-parallel-{topology}{{tile-size={candidate['tile_sizes']['BK']}}}",
                "materialize-numeric-reduction", "fuse-reduction-consumer",
            ]
        if label != "identity" and not pipeline:
            records[label] = {"pass": False, "reason": "missing_executable_witness",
                              "graph_id": candidate["graph_id"]}
            continue
        ir_path = output_dir / f"{label}.mlir"
        facts_path = output_dir / f"{label}.facts.json"
        command = [str(optimizer), str(source), "--verify-each"]
        command.extend(_pass_flag(action) for action in pipeline)
        command += [f"--extract-joint-task-graph-facts=output={facts_path}",
                    "-o", str(ir_path)]
        result = _run(command)
        comparison: dict[str, Any] | None = None
        if result["returncode"] == 0 and facts_path.is_file():
            try:
                comparison = compare_graph_facts_by_semantic_signature(
                    candidate, json.loads(facts_path.read_text(encoding="utf-8")))
            except (GraphFactsMismatch, ValueError) as error:
                comparison = {"pass": False, "reason": str(error)}
        records[label] = {"graph_id": candidate["graph_id"],
                          "witness": witness, "verifier": result,
                          "replay_pipeline": pipeline,
                          "manifest_witness_executable": witness["executable"],
                          "replay_uses_manifest_witness":
                          pipeline == witness["pass_pipeline"],
                          "cpp_graph_facts_comparison": comparison,
                          "ir_path": str(ir_path) if ir_path.is_file() else None,
                          "facts_path": str(facts_path) if facts_path.is_file() else None,
                          "pass": result["returncode"] == 0 and
                          ir_path.is_file() and facts_path.is_file() and
                          comparison is not None and comparison["pass"]}
    return {"schema": "orbit-joint-standalone-pass-regression-v1",
            "source_module_sha256": manifest["source_module_sha256"],
            "representatives": records,
            "pass": all(record["pass"] for record in records.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.manifest, args.amoeba_opt, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
