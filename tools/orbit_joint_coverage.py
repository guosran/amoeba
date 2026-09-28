#!/usr/bin/env python3
"""Read-only inventory scan for source-bound joint graph rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TASK = re.compile(r"(?P<result>%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                  r"taskflow\.task\s+@(?P<name>[A-Za-z_][A-Za-z0-9_.]*)")
READS = re.compile(r"will_reads\((.*?)\)", re.S)
SSA = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_path(value: str) -> Path:
    return Path(value.replace("/home/shiran/", "/home/x/shiran/", 1))


def scan_taskflow_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "available": False,
                "status": "source_identity_unproven"}
    source = path.read_text(encoding="utf-8")
    matches = list(TASK.finditer(source))
    producers: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        read = READS.search(source[match.end():end])
        if read is not None:
            for operand in SSA.findall(read.group(1).split(":", 1)[0]):
                if operand in producers:
                    edges.add((producers[operand], match.group("name")))
        producers[match.group("result")] = match.group("name")
    return {"path": str(path), "available": True,
            "source_sha256": _sha(path),
            "taskflow_task_count": len(matches),
            "raw_ssa_producer_consumer_edges": len(edges),
            "raw_ssa_edges": [list(edge) for edge in sorted(edges)],
            "status": "source_identity_unproven"}


def scan(saved_manifest: Path, source_inventory: Path,
         polybench_screen: Path) -> dict[str, Any]:
    manifest = json.loads(saved_manifest.read_text())
    inventory = json.loads(source_inventory.read_text())
    polybench = json.loads(polybench_screen.read_text())
    rows = []
    for workload in manifest["workloads"]:
        path = _local_path(workload["source_path"])
        row = scan_taskflow_source(path)
        row.update({"name": workload["name"],
                    "declared_task_count": workload["task_count"],
                    "source_hash_matches_manifest": (
                        row.get("source_sha256") == workload["source_sha256"]),
                    "proven_fusion_eligible_pairs": 0,
                    "proven_tiling_variants": 0,
                    "proven_tile_local_fusion_opportunities": 0,
                    "proven_sequential_k_consumer_fusion_opportunities": 0,
                    "proven_parallel_reduction_consumer_fusion_opportunities": 0,
                    "external_user_rejections": "unclassified",
                    "alias_side_effect_rejections": "unclassified",
                    "unknown_access_rejections": "unclassified"})
        rows.append(row)
    for name in ("cnn", "fft", "plain_gemm_f32"):
        path = _local_path(inventory["source_examples"][name])
        row = scan_taskflow_source(path)
        row.update({"name": "source_example_" + name,
                    "proven_fusion_eligible_pairs": 0,
                    "proven_tiling_variants": 0,
                    "proven_tile_local_fusion_opportunities": 0,
                    "proven_sequential_k_consumer_fusion_opportunities": 0,
                    "proven_parallel_reduction_consumer_fusion_opportunities": 0})
        rows.append(row)
    return {
        "schema": "orbit-whole-workload-joint-rewrite-coverage-v1",
        "scope": "read_only_source_graph_scan",
        "saved_manifest_sha256": _sha(saved_manifest),
        "source_inventory_sha256": _sha(source_inventory),
        "polybench_screen_sha256": _sha(polybench_screen),
        "saved_and_example_sources": rows,
        "polybench": {"screened_kernels": polybench["kernel_count"],
                      "dense_product_stages": polybench["dense_product_stage_count"],
                      "proven_joint_pipeline_stages": 0,
                      "status": "source_identity_unproven"},
        "proven_joint_pipeline_count": 0,
        "reason": "no scanned workload is proven equivalent to the canonical i32 producer-consumer source with exact alias, access, and user contracts",
        "performance_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-manifest", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--polybench-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.saved_manifest, args.source_inventory,
                  args.polybench_screen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
