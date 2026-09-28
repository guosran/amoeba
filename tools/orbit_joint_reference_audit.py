#!/usr/bin/env python3
"""Independent numeric and legality audit for the C++ joint rewrite passes.

The Python graph model is a test oracle.  It does not publish ORBIT's
production candidates after the MLIR-pass ownership change.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from orbit_joint_pipeline import canonical_pipeline_identity, reference_pipeline
from orbit_joint_rewrite_space import (ACTION_KINDS, REJECTION_CODES,
                                       apply_action, canonical_graph_id,
                                       enumerate_rewrite_dag, interpret_graph,
                                       root_graph, validate_graph)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _cases() -> list[tuple[str, dict[str, list[int]]]]:
    rng = random.Random(47291)
    return [
        ("positive_nonzero_C_and_E", {
            "A": [i % 7 + 1 for i in range(64)],
            "B": [i % 5 + 1 for i in range(64)],
            "C_initial": [i % 11 + 1 for i in range(64)],
            "E": [i % 13 + 1 for i in range(64)]}),
        ("signed_zero", {
            "A": [((i * 7) % 17) - 8 for i in range(64)],
            "B": [0 if i % 5 == 0 else ((i * 11) % 19) - 9
                  for i in range(64)],
            "C_initial": [((i * 3) % 13) - 6 for i in range(64)],
            "E": [((i * 5) % 15) - 7 for i in range(64)]}),
        ("overflow", {
            "A": [(1 << 31) - 1 if i % 2 else -(1 << 31)
                  for i in range(64)],
            "B": [-3 if i % 2 else (1 << 31) - 1 for i in range(64)],
            "C_initial": [-7 if i % 2 else (1 << 31) - 1
                          for i in range(64)],
            "E": [0x7fffffff if i % 3 else -0x80000000
                  for i in range(64)]}),
        ("fixed_seed_47291", {
            name: [rng.randint(-100000, 100000) for _ in range(64)]
            for name in ("A", "B", "C_initial", "E")}),
    ]


def _expected_partial(values: Mapping[str, Sequence[int]],
                      m_range: Sequence[int], n_range: Sequence[int],
                      k_range: Sequence[int]) -> list[int]:
    result: list[int] = []
    for m in range(*m_range):
        for n in range(*n_range):
            accumulator = 0
            for k in range(*k_range):
                accumulator = (accumulator + values["A"][m * 8 + k] *
                               values["B"][k * 8 + n]) & 0xffffffff
            result.append(accumulator)
    return result


def _tile(values: Sequence[int], m_range: Sequence[int],
          n_range: Sequence[int]) -> list[int]:
    return [values[m * 8 + n] & 0xffffffff
            for m in range(*m_range) for n in range(*n_range)]


def _numeric(identity: Mapping[str, Any], graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for name, values in _cases():
        expected = reference_pipeline(identity, values)
        checks = {"T": 0, "D": 0, "state_carry": 0,
                  "private_partial": 0, "reduction_output": 0}
        for graph in graphs:
            result = interpret_graph(graph, values)
            if result["T"] != expected["T"] or result["D"] != expected["D"]:
                raise ValueError("numeric mismatch for " + name + " " +
                                 canonical_graph_id(graph))
            checks["T"] += 1
            checks["D"] += 1
            for m_range in graph["m_tiles"]:
                for n_range in graph["n_tiles"]:
                    tile_id = f"tile_m{m_range[0]}_n{n_range[0]}"
                    final_t = _tile(expected["T"], m_range, n_range)
                    if graph["k_policy"] == "sequential":
                        chain = result["state_carry"][tile_id]
                        if chain[0] != _tile(values["C_initial"], m_range, n_range) or \
                                chain[-1] != final_t or \
                                len(chain) != len(graph["k_blocks"]) + 1:
                            raise ValueError("state chain mismatch " + tile_id)
                        checks["state_carry"] += len(chain)
                    else:
                        partials = result["partials"][tile_id]
                        for index, block in enumerate(graph["k_blocks"]):
                            kid = f"k{index}_{block[0]}_{block[1]}"
                            if partials[kid] != _expected_partial(
                                    values, m_range, n_range, block):
                                raise ValueError("private partial mismatch " + kid)
                            checks["private_partial"] += 1
                        if result["reduction_outputs"][tile_id] != final_t:
                            raise ValueError("reduction output mismatch " + tile_id)
                        checks["reduction_output"] += 1
        records.append({"case": name, "complete_graphs_checked": len(graphs),
                        "reference_T_sha256": _sha(expected["T"]),
                        "reference_D_sha256": _sha(expected["D"]),
                        "debug_checks": checks, "pass": True})
    return {"schema": "orbit-joint-reference-numeric-audit-v1",
            "role": "independent_reference_oracle",
            "arithmetic": "i32_modulo_2^32", "all_pass": True,
            "cases": records,
            "candidate_case_comparisons": len(records) * len(graphs)}


def _legality(identity: Mapping[str, Any]) -> dict[str, Any]:
    negative = [
        ("external_user", "intermediate_external_users", ["external"],
         "EXTERNAL_USER_PRESENT"),
        ("alias", "intermediate_aliases", ["E"], "ALIAS_PROOF_MISSING"),
        ("producer_side_effect", "producer_side_effect_free", False,
         "SIDE_EFFECT_UNPROVEN"),
        ("consumer_side_effect", "consumer_side_effect_free", False,
         "SIDE_EFFECT_UNPROVEN"),
        ("unknown_access", "consumer_unknown_access", True, "UNKNOWN_ACCESS"),
        ("multiple_exits", "multiple_exits", True, "MULTIPLE_EXITS"),
        ("dynamic_control", "dynamic_control", True, "DYNAMIC_CONTROL"),
        ("region_mismatch", "producer_consumer_region", "mismatch",
         "REGION_MISMATCH"),
        ("halo", "consumer_halo", True, "HALO_UNSUPPORTED"),
        ("broadcast", "consumer_broadcast", True, "BROADCAST_UNSUPPORTED"),
        ("reshape", "consumer_reshape", True, "RESHAPE_UNSUPPORTED"),
        ("reduction_consumer", "consumer_is_reduction", True,
         "REDUCTION_CONSUMER_UNSUPPORTED"),
    ]
    records = []
    for name, key, value, expected in negative:
        graph = root_graph(identity, contract_overrides={key: value})
        result = apply_action(graph, {"kind": "FuseProducerConsumer"})
        if result.accepted or result.reason_code != expected or result.graph is not None:
            raise ValueError("legality rejection mismatch for " + name)
        records.append({"fixture": name, "expected_reason_code": expected,
                        "actual_reason_code": result.reason_code,
                        "atomic_rejection": True, "pass": True})
    return {"schema": "orbit-joint-reference-legality-audit-v1",
            "role": "independent_reference_oracle",
            "rejection_codes": list(REJECTION_CODES),
            "negative_fixtures": records, "all_pass": True}


def run(source: Path, output: Path) -> dict[str, Any]:
    identity = canonical_pipeline_identity(source)
    graph = root_graph(identity)
    dag = enumerate_rewrite_dag(graph, max_depth=8, max_actions=2000,
                                tile_m=4, tile_n=4, tile_k=4)
    if dag["statistics"]["budget_exhausted"]:
        raise ValueError("reference DAG action budget exhausted")
    candidates = dag["candidate_graphs"]
    if any(not validate_graph(item) for item in candidates):
        raise ValueError("reference DAG published an incomplete graph")
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "numeric_equivalence_results.json", _numeric(identity, candidates))
    _write(output / "fusion_legality_audit.json", _legality(identity))
    _write(output / "reference_rewrite_dag_statistics.json", {
        "schema": dag["schema"], "role": "independent_reference_oracle",
        "statistics": dag["statistics"], "action_kinds": list(ACTION_KINDS),
        "candidate_graph_ids": [canonical_graph_id(item) for item in candidates],
        "rejection_reason_histogram": dict(Counter(
            item["reason_code"] for item in dag["rejected_transitions"])),
    })
    _write(output / "intermediate_lifetime_audit.json", {
        "schema": "orbit-joint-reference-lifetime-audit-v1",
        "role": "independent_reference_oracle",
        "candidate_count": len(candidates),
        "visibility_counts": dict(Counter(
            item["intermediate"]["visibility"] for item in candidates)),
        "private_partial_graph_count": sum(
            bool(item["intermediate"]["partial_buffers"]) for item in candidates),
        "fused_external_T_allocations": sum(
            item["intermediate"]["allocation"] != "eliminated"
            for item in candidates if item["mode_tags"]["fused"]),
        "completion_joins_numeric": False,
        "all_pass": True,
    })
    return {"source_id": identity["canonical_pipeline_id"],
            "reference_candidate_count": len(candidates),
            "numeric_cases": len(_cases()), "pass": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.input, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
