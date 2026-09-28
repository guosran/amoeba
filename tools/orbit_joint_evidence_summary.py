#!/usr/bin/env python3
"""Summarize C++ MLIR joint rewrite evidence without changing candidates.

All legality, graph identities, witnesses, and Taskflow rewrites originate in
the C++ passes.  This utility only copies and indexes their observed output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from orbit_joint_pipeline import canonical_pipeline_identity


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def run(source: Path, manifest_path: Path, regression_path: Path,
        evidence_dir: Path, optimizer: Path, worktree: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("production C++ rewrite manifest is incomplete")
    candidates = manifest["candidates"]
    transitions = manifest["transitions"]
    rejections = manifest["rejections"]
    stats = manifest["statistics"]
    evidence_dir.mkdir(parents=True, exist_ok=True)

    source_identity = canonical_pipeline_identity(source)
    source_identity["identity_authority"] = "canonical_source_and_cpp_mlir_pass"
    source_identity["source_module_sha256_raw_bytes"] = _sha(source)
    source_identity["source_module_sha256_normalized_mlir_print"] = (
        manifest["source_module_sha256"])
    _write(evidence_dir / "canonical_pipeline_identity.json", source_identity)

    _write(evidence_dir / "source_to_joint_variant_provenance.json", {
        "schema": "orbit-joint-source-variant-provenance-v1",
        "canonical_source_id": manifest["canonical_source_id"],
        "source_module_sha256_kind": manifest["source_module_sha256_kind"],
        "source_module_sha256_raw_bytes": _sha(source),
        "candidate_count": len(candidates),
        "variants": [{"graph_id": item["graph_id"],
                      "tile_sizes": item["tile_sizes"],
                      "k_policy": item["k_policy"],
                      "reduction_topology": item["reduction_topology"],
                      "fusion": item["fusion"],
                      "witness": item["witness"]} for item in candidates],
    })
    _write(evidence_dir / "joint_variant_schema.json", {
        "schema": "orbit-joint-variant-schema-v1",
        "production_owner": "C++ MLIR enumerate-joint-semantic-rewrites pass",
        "manifest_schema": manifest["schema"],
        "candidate_graph_schema": sorted({c["schema"] for c in candidates}),
        "candidate_fields": sorted(candidates[0]) if candidates else [],
        "variant_class": "semantic_graph_candidate",
        "fusion_layers": {"task_graph_fusion": True,
                          "kernel_dfg_fusion": False,
                          "fu_fusion": False,
                          "streaming_memory_fusion": False},
        "excluded_identity_inputs": ["action_history", "runtime_values", "backend",
                                     "mapping", "cost", "placement", "replication"],
    })

    action_kinds = sorted({item["kind"] for item in manifest["action_catalog"]})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in transitions + rejections:
        grouped[item["action"]["kind"]].append(item)
    _write(evidence_dir / "rewrite_action_contract.json", {
        "schema": "orbit-joint-action-contract-v1",
        "owner": "C++ MLIR pass",
        "action_catalog": manifest["action_catalog"],
        "transition_fields": sorted(transitions[0]) if transitions else [],
        "rejection_fields": sorted(rejections[0]) if rejections else [],
        "atomic_manifest_publication": True,
        "candidate_count": len(candidates),
        "transition_count": len(transitions),
        "rejection_count": len(rejections),
    })
    _write(evidence_dir / "action_prerequisite_matrix.json", {
        "schema": "orbit-joint-action-prerequisites-v1",
        "actions": {kind: sorted({requirement for item in grouped[kind]
                                  for requirement in item.get("prerequisites", [])})
                    for kind in action_kinds},
    })
    _write(evidence_dir / "action_conflict_matrix.json", {
        "schema": "orbit-joint-action-conflicts-v1",
        "actions": {kind: {
            "conflicts": sorted({conflict for item in grouped[kind]
                                 for conflict in item.get("conflicts", [])}),
            "rejection_codes": dict(sorted(Counter(
                item["reason_code"] for item in grouped[kind]
                if item.get("accepted") is False and item.get("reason_code")
            ).items())),
        } for kind in action_kinds},
    })
    _write(evidence_dir / "rewrite_dag_statistics.json", {
        "schema": "orbit-joint-rewrite-dag-statistics-v1",
        "production_owner": "C++ MLIR pass",
        "source_module_sha256": manifest["source_module_sha256"],
        "statistics": stats,
        "complete_manifest": manifest["complete"],
        "mode_counts": dict(sorted(Counter(
            c["mode_tags"]["primary"] for c in candidates).items())),
        "k_policy_counts": dict(sorted(Counter(
            c["k_policy"] for c in candidates).items())),
        "reduction_topology_counts": dict(sorted(Counter(
            c["reduction_topology"] for c in candidates).items())),
        "executable_witnesses": sum(bool(c["witness"]["executable"])
                                    for c in candidates),
    })
    duplicate_examples = [{"source_graph_id": item["source_graph_id"],
                           "action": item["action"],
                           "result_graph_id": item["result_graph_id"]}
                          for item in transitions if item.get("duplicate_state")][:8]
    _write(evidence_dir / "canonical_graph_identity_contract.json", {
        "schema": "orbit-joint-canonical-graph-identity-v1",
        "owner": "C++ MLIR pass",
        "source_module_sha256": manifest["source_module_sha256"],
        "candidate_graph_ids_unique": len({c["graph_id"] for c in candidates})
                                      == len(candidates),
        "identity_excludes_action_history": True,
        "duplicate_states_removed": stats["duplicate_states_removed"],
        "duplicate_transition_examples": duplicate_examples,
        "semantic_identity_inputs": ["source", "task_roles", "task_bodies",
                                     "M_N_K_ranges", "typed_edges", "regions",
                                     "K_policy", "reduction_topology",
                                     "fusion_groups", "intermediate_lifetime",
                                     "completion_join", "numeric_reduction",
                                     "arithmetic_semantics"],
    })

    records = regression["representatives"]
    _write(evidence_dir / "taskflow_materialization_results.json", {
        "schema": "orbit-joint-taskflow-materialization-results-v1",
        "production_owner": "standalone C++ MLIR passes",
        "representative_count": len(records),
        "verified_count": sum(v["verifier"]["returncode"] == 0
                              for v in records.values() if "verifier" in v),
        "representatives": {name: {
            "graph_id": value["graph_id"],
            "verifier_pass": value.get("verifier", {}).get("returncode") == 0,
            "replay_pipeline": value.get("replay_pipeline"),
            "manifest_witness_executable": value.get(
                "manifest_witness_executable"),
            "replay_uses_manifest_witness": value.get(
                "replay_uses_manifest_witness"),
            "ir_path": value.get("ir_path"),
        } for name, value in records.items()},
    })
    _write(evidence_dir / "cpp_graph_facts_comparison.json", {
        "schema": "orbit-joint-cpp-facts-comparison-v1",
        "bijection_basis": "semantic_role_k_interval_output_region",
        "pass_count": sum(value["pass"] for value in records.values()),
        "representative_count": len(records),
        "representatives": {name: value.get("cpp_graph_facts_comparison")
                            for name, value in records.items()},
    })
    _write(evidence_dir / "version_manifest.json", {
        "schema": "orbit-joint-version-manifest-v1",
        "source_worktree": str(worktree),
        "source_head": subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        "optimizer_path": str(optimizer),
        "optimizer_sha256": _sha(optimizer),
        "python": _version(["python3", "--version"]),
        "cmake": _version(["cmake", "--version"]),
        "clang_format": _version(["/home/x/shiran/llvm-project/build/bin/clang-format", "--version"]),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.manifest, args.regression, args.evidence_dir,
        args.amoeba_opt, args.worktree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
