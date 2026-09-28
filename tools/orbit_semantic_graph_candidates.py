#!/usr/bin/env python3
"""Source-bound semantic GEMM graph candidates for the ORBIT program entry.

The semantic enumerator owns M/N/K rules. This module publishes its variants
as graph candidate actions and materializes selected Taskflow IR. It never
asks a hardware backend for legality or cost.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Sequence

from orbit_task_variant_space import (canonical_program_identity,
                                      enumeration_statistics, enumerate_variants,
                                      interpret_variant,
                                      reference, validate_variant)


SCHEMA = "amoeba-semantic-gemm-graph-candidates-v1"
OPT = Path("/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/mlir-amoeba-opt")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_set(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    require(result and all(item > 0 for item in result), "tile factors must be positive")
    return result


def _verify_mlir(path: Path, optimizer: Path) -> tuple[bool, str]:
    result = subprocess.run([str(optimizer), str(path), "--verify-each",
                             "-o", "/dev/null"], text=True, capture_output=True)
    return result.returncode == 0, result.stderr[-2000:]


def _fact_record(program: dict, variant: dict, index: int) -> dict:
    graph = {"tasks": variant["tasks"], "typed_edges": variant["edges"],
             "output_tiles": variant["output_tiles"],
             "k_blocks": variant["k_blocks"],
             "completion_task_id": variant["completion_task_id"]}
    action = {"kind": "semantic_gemm_task_tiling", "BM": variant["BM"],
              "BN": variant["BN"], "BK": variant["BK"],
              "k_policy": variant["k_policy"],
              "reduction_topology": variant["reduction_topology"],
              "tail_policy": variant["tail_policy"]}
    return {"schema": SCHEMA, "record_type": "variant", "index": index,
            "canonical_program_id": program["canonical_program_id"],
            "graph_variant_id": variant["variant_id"],
            "semantic_variant_id": variant["variant_id"],
            "source_module_sha256": program["source_program_sha256"],
            "action": action, "graph_facts_sha256": digest(graph),
            "task_count": len(variant["tasks"]),
            "compute_task_count": sum(t["task_kind"] == "gemm" for t in variant["tasks"]),
            "reduction_task_count": sum(t["task_kind"] == "reduction" for t in variant["tasks"]),
            "typed_edge_count": len(variant["edges"]),
            "semantic_legal": True, "backend_status": "unknown",
            "variant_space_index": index}


def _numeric_cases(program: dict) -> list[tuple[str, dict]]:
    m, n, k = program["M"], program["N"], program["K"]
    cases = [
        ("positive", {"A": [i % 7 + 1 for i in range(m*k)],
                      "B": [i % 5 + 1 for i in range(k*n)],
                      "C_initial": [i % 3 + 1 for i in range(m*n)]}),
        ("signed_zero", {"A": [((i * 7) % 17) - 8 for i in range(m*k)],
                          "B": [0 if i % 5 == 0 else ((i * 11) % 19) - 9
                                for i in range(k*n)],
                          "C_initial": [((i * 3) % 13) - 6 for i in range(m*n)]}),
        ("overflow", {"A": [(1 << 31)-1 if i % 2 else -(1 << 31)
                              for i in range(m*k)],
                      "B": [-3 if i % 2 else (1 << 31)-1 for i in range(k*n)],
                      "C_initial": [-7 if i % 2 else (1 << 31)-1
                                    for i in range(m*n)]}),
    ]
    rng = random.Random(47291)
    cases.append(("fixed_seed_47291", {
        "A": [rng.randint(-100, 100) for _ in range(m*k)],
        "B": [rng.randint(-100, 100) for _ in range(k*n)],
        "C_initial": [rng.randint(-100, 100) for _ in range(m*n)]}))
    return cases


def _representatives(variants: list[dict], program: dict) -> list[dict]:
    m, n, k = (program[key] for key in ("M", "N", "K"))
    targets = [
        ("m_only", m//2, n, k, "sequential_accumulate", "none"),
        ("n_only", m, n//2, k, "sequential_accumulate", "none"),
        ("m_n", m//2, n//2, k, "sequential_accumulate", "none"),
        ("sequential_k", m, n, k//2, "sequential_accumulate", "none"),
        ("parallel_linear", m, n, k//2, "parallel_partial_reduce", "linear"),
        ("parallel_tree", m, n, k//2, "parallel_partial_reduce", "balanced_tree"),
        ("m_n_k", m//2, n//2, k//2, "parallel_partial_reduce", "balanced_tree"),
    ]
    rows = []
    for name, bm, bn, bk, policy, topology in targets:
        match = next((v for v in variants if (v["BM"], v["BN"], v["BK"],
                     v["k_policy"], v["reduction_topology"]) ==
                     (bm, bn, bk, policy, topology)), None)
        if match is not None:
            rows.append({"name": name, "variant": match})
    return rows


def _materialize(program: dict, variant: dict, destination: Path,
                 optimizer: Path) -> dict:
    from orbit_semantic_taskflow import materialize_taskflow
    text = materialize_taskflow(program, variant)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    valid, diagnostic = _verify_mlir(destination, optimizer)
    fact_path = destination.with_suffix(".facts.json")
    fact_result = None
    if valid:
        fact_result = subprocess.run([
            str(optimizer), str(destination), "--verify-each",
            f"--extract-joint-task-graph-facts=output={fact_path} max-tile-factor=8",
            "-o", "/dev/null"], text=True, capture_output=True)
    fact_edges_match = False
    unexpected_fact_edges: list[list[str]] = []
    missing_fact_edges: list[list[str]] = []
    if fact_result is not None and fact_result.returncode == 0 and fact_path.is_file():
        facts = json.loads(fact_path.read_text())
        actual = {(edge["source_task"], edge["target_task"])
                  for edge in facts["dependencies"]}
        expected = {(edge["source_task"], edge["target_task"])
                    for edge in variant["edges"]
                    if edge["kind"] != "completion_only"}
        unexpected_fact_edges = [list(edge) for edge in sorted(actual - expected)]
        missing_fact_edges = [list(edge) for edge in sorted(expected - actual)]
        fact_edges_match = not unexpected_fact_edges and not missing_fact_edges
    return {"variant_id": variant["variant_id"], "ir_path": str(destination),
            "ir_sha256": sha_file(destination), "ir_verifier_pass": valid,
            "ir_verifier_diagnostic": diagnostic if not valid else None,
            "cxx_facts_pass": fact_result is not None and fact_result.returncode == 0,
            "cxx_fact_edges_match_semantic": fact_edges_match,
            "cxx_unexpected_edges": unexpected_fact_edges,
            "cxx_missing_edges": missing_fact_edges,
            "cxx_facts_path": str(fact_path) if fact_path.is_file() else None,
            "cxx_facts_diagnostic": (fact_result.stderr[-2000:] if fact_result and
                                     fact_result.returncode else None),
            "source_program_sha256": program["source_program_sha256"],
            "canonical_program_id": program["canonical_program_id"]}


def run(args: argparse.Namespace) -> dict:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    valid, diagnostic = _verify_mlir(source, args.amoeba_opt)
    require(valid, "canonical MLIR verifier failed: " + diagnostic)
    program = canonical_program_identity(source)
    bm_set = parse_set(args.bm_set)
    bn_set = parse_set(args.bn_set)
    bk_set = parse_set(args.bk_set)
    policies = args.k_policies.split(",")
    topologies = args.reduction_topologies.split(",")
    statistics = enumeration_statistics(
        program, bm_set, bn_set, bk_set, policies,
        tail_policy=args.tail_policy, reduction_topologies=topologies)
    variants = enumerate_variants(
        program, bm_set, bn_set, bk_set, policies,
        tail_policy=args.tail_policy,
        reduction_topologies=topologies)
    require(variants, "no semantically legal variants: " +
            json.dumps(statistics["rejection_reason_histogram"], sort_keys=True))
    require(statistics["legal_raw_combination_count"] >= len(variants),
            "enumeration statistics disagree with unique candidates")
    for variant in variants:
        validate_variant(program, variant)
    ids = [variant["variant_id"] for variant in variants]
    require(len(ids) == len(set(ids)), "semantic variant ID collision")
    records = [_fact_record(program, variant, index)
               for index, variant in enumerate(variants)]
    by_id = {variant["variant_id"]: variant for variant in variants}
    header = {"schema": SCHEMA, "record_type": "header", "complete": True,
              "source_module_sha256": program["source_program_sha256"],
              "canonical_program_id": program["canonical_program_id"],
              "owner": "tools/orbit_task_variant_space.py",
              "candidate_action_kind": "semantic_gemm_task_tiling",
              "backend_status": "unknown"}
    footer = {"schema": SCHEMA, "record_type": "footer", "complete": True,
              "accepted_count": len(variants),
              "rejected_count": statistics["rejected_count"],
              "deduplicated_count": statistics["legal_raw_combination_count"] - len(variants),
              "rejection_reason_histogram": statistics["rejection_reason_histogram"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "canonical_program_identity.json", program)
    (output / "semantic_variant_space.json").write_bytes(canonical(
        {"schema": "orbit-semantic-variant-space-v2",
         "canonical_program_id": program["canonical_program_id"],
         "count": len(variants), "variants": variants}) + b"\n")
    (output / "graph_candidate_manifest.jsonl").write_text(
        "".join(canonical(row).decode() + "\n" for row in [header, *records, footer]))
    write_json(output / "source_to_task_variant_provenance.json",
               {"schema": "orbit-source-to-task-variant-v1",
                "source_path": str(source), "source_module_sha256": sha_file(source),
                "canonical_program_id": program["canonical_program_id"],
                "normalized_access_sha256": program["operation_access_sha256"],
                "variant_ids": ids, "graph_candidate_manifest_sha256":
                sha_file(output / "graph_candidate_manifest.jsonl")})
    write_json(output / "semantic_variant_schema.json",
               {"schema": "orbit-semantic-variant-schema-v2",
                "canonical_owner": "tools/orbit_task_variant_space.py",
                "dimensions": ["BM", "BN", "BK", "k_policy", "reduction_topology"],
                "region": "half-open lower/upper, compatible with TaskEdgeContract",
                "edge_scopes": ["tile_local", "tensor_wide"],
                "edge_roles": ["state_carry", "reduction_input", "completion_only"],
                "semantic_tail_supported": True,
                "backend_tail_supported": "unknown",
                "backend_status": "unknown"})
    write_json(output / "variant_identity_contract.json",
               {"schema": "orbit-semantic-identity-v2",
                "canonical_program_id": program["canonical_program_id"],
                "unique_count": len(ids), "runtime_inputs_excluded": True,
                "hardware_fields_excluded": True,
                "topology_included": True})
    categories = Counter()
    for variant in variants:
        split = tuple(variant[key] < program[axis] for key, axis in
                      (("BM", "M"), ("BN", "N"), ("BK", "K")))
        categories["".join(axis for axis, yes in zip("MNK", split) if yes) or "identity"] += 1
    write_json(output / "axis_enumeration_summary.json",
               {"schema": "orbit-axis-enumeration-v1", "BM": bm_set,
                "BN": bn_set, "BK": bk_set,
                "k_policies": policies,
                "reduction_topologies": topologies,
                "enumeration_statistics": statistics,
                "variant_count": len(variants), "category_counts": dict(categories),
                "identity_present": any(v["BM"] == program["M"] and
                                        v["BN"] == program["N"] and
                                        v["BK"] == program["K"] for v in variants)})
    write_json(output / "region_coverage_audit.json",
               {"schema": "orbit-region-coverage-v2", "checked_variants": len(variants),
                "all_exact": True, "output_elements": program["M"]*program["N"],
                "K_indices_per_output": program["K"]})
    write_json(output / "tail_legality_audit.json",
               {"schema": "orbit-tail-legality-v1", "tail_policy": args.tail_policy,
                "semantic_tail_supported": True,
                "backend_tail_supported": "unknown",
                "tail_required_variant_count":
                sum(variant["tail_required"] for variant in variants),
                "validated_variant_count": len(variants)})
    seq = [v for v in variants if v["k_policy"] == "sequential_accumulate"]
    par = [v for v in variants if v["k_policy"] == "parallel_partial_reduce"]
    write_json(output / "sequential_k_audit.json",
               {"schema": "orbit-sequential-k-audit-v1", "variant_count": len(seq),
                "state_carry_edges": sum(e["kind"] == "state_carry" for v in seq
                                         for e in v["edges"]),
                "validated": True})
    write_json(output / "parallel_reduction_audit.json",
               {"schema": "orbit-parallel-reduction-audit-v1",
                "variant_count": len(par),
                "topology_counts": dict(Counter(v["reduction_topology"] for v in par)),
                "reduction_task_count": sum(t["task_kind"] == "reduction" for v in par
                                            for t in v["tasks"]),
                "validated": True})
    numeric = []
    if program["dtype"] == "i32":
        for name, values in _numeric_cases(program):
            expected = reference(program, values)
            count = 0
            for variant in variants:
                require(interpret_variant(program, variant, values) == expected,
                        "numeric mismatch: " + name + " " + variant["variant_id"])
                count += 1
            numeric.append({"case": name, "checked_variants": count,
                            "expected_C_sha256": digest(expected), "all_exact": True})
    write_json(output / "numeric_equivalence_results.json",
               {"schema": "orbit-semantic-numeric-equivalence-v2",
                "arithmetic": ("i32 modulo 2^32" if program["dtype"] == "i32"
                               else "f32 source order; reassociation disallowed"),
                "cases": numeric, "all_exact": bool(numeric),
                "status": ("checked" if numeric else "not_evaluated_for_f32")})
    materialized = []
    for item in _representatives(variants, program):
        destination = output / "materialized" / (item["name"] + ".mlir")
        try:
            row = _materialize(program, item["variant"], destination,
                               args.amoeba_opt)
        except Exception as error:
            row = {"variant_id": item["variant"]["variant_id"],
                   "ir_verifier_pass": False, "cxx_facts_pass": False,
                   "failure_reason": f"{type(error).__name__}: {error}"}
        materialized.append({"name": item["name"], **row})
    default = next((v for v in variants if v["BM"] == program["M"] and
                    v["BN"] == program["N"] and v["BK"] == program["K"] and
                    v["k_policy"] == "sequential_accumulate"), variants[0])
    selected_id = args.select_variant_id or default["variant_id"]
    require(selected_id in by_id, "selected semantic variant ID not in graph manifest")
    try:
        selected = _materialize(program, by_id[selected_id],
                                output / "selected_taskflow.mlir", args.amoeba_opt)
    except Exception as error:
        selected = {"variant_id": selected_id, "ir_verifier_pass": False,
                    "cxx_facts_pass": False,
                    "failure_reason": f"{type(error).__name__}: {error}"}
    write_json(output / "taskflow_materialization_results.json",
               {"schema": "orbit-taskflow-materialization-v1",
                "representatives": materialized, "selected": selected,
                "selected_graph_variant_id": selected_id,
                "selected_action": next(row["action"] for row in records
                                        if row["graph_variant_id"] == selected_id),
                "source_module_sha256": program["source_program_sha256"]})
    screen = (json.loads(args.polybench_screen.read_text())
              if args.polybench_screen is not None else None)
    write_json(output / "whole_workload_semantic_coverage.json",
               {"schema": "orbit-whole-workload-semantic-coverage-v2",
                "canonical": {"proven_gemm": 1, "semantic_variants": len(variants)},
                "polybench": {"screened_kernels": screen["kernel_count"] if screen else 0,
                              "dense_product_source_stages":
                              screen["dense_product_stage_count"] if screen else 0,
                              "proven_source_bound_semantic_stages": 0,
                              "status": "source_extraction_and_alias_contract_unproven",
                              "screen_sha256": (sha_file(args.polybench_screen)
                                                if args.polybench_screen else None)},
                "saved_cnn_gpt_qwen_fft": {"status": "canonical_source_semantics_unproven",
                                      "proven_source_bound_semantic_programs": 0,
                                      "inventory_sha256":
                                      (sha_file(args.saved_workload_inventory)
                                       if args.saved_workload_inventory else None)},
                "backend_status": "unknown", "performance_claim": False})
    return {"schema": SCHEMA, "canonical_program_id": program["canonical_program_id"],
            "semantic_variant_count": len(variants),
            "selected_graph_variant_id": selected_id,
            "selected_ir_verifier_pass": selected["ir_verifier_pass"],
            "selected_cxx_facts_pass": selected["cxx_facts_pass"],
            "selected_cxx_fact_edges_match_semantic":
            selected.get("cxx_fact_edges_match_semantic", False),
            "representative_ir_verifier_pass_count": sum(row["ir_verifier_pass"]
                                                       for row in materialized),
            "representative_cxx_fact_edge_match_count":
            sum(row.get("cxx_fact_edges_match_semantic", False)
                for row in materialized)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, default=OPT)
    parser.add_argument("--bm-set", default="1,2,4,8")
    parser.add_argument("--bn-set", default="1,2,4,8")
    parser.add_argument("--bk-set", default="1,2,4,8")
    parser.add_argument("--k-policies", default="sequential_accumulate,parallel_partial_reduce")
    parser.add_argument("--reduction-topologies", default="none,linear,balanced_tree")
    parser.add_argument("--tail-policy", choices=("reject", "exact"), default="reject")
    parser.add_argument("--select-variant-id", default="")
    parser.add_argument("--polybench-screen", type=Path)
    parser.add_argument("--saved-workload-inventory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(json.dumps(run(args), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"semantic graph candidates failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
