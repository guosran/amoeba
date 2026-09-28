#!/usr/bin/env python3
"""End-to-end checks for the source-bound semantic graph entry."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from orbit_semantic_taskflow import materialize_taskflow  # noqa: E402
from orbit_task_variant_space import (canonical_program_identity,
                                      enumerate_variants)  # noqa: E402
from test_orbit_task_variant_space import _descriptor  # noqa: E402
OPT = Path("/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/mlir-amoeba-opt")
SOURCE = Path(
    "/tmp/amoeba-joint-scheduling-review/orbit_vectorcgra_memory_blocking_review/"
    "orbit_generic_blocking_mainline/gemm_8x8_i32.mlir"
)


@unittest.skipUnless(OPT.is_file() and SOURCE.is_file(),
                     "local canonical GEMM and Taskflow optimizer required")
class SemanticGraphCandidateTest(unittest.TestCase):
    def test_nondivisible_tail_rejection_is_deterministic_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="orbit-semantic-reject-test-") as temporary:
            output = Path(temporary) / "rejected"
            result = subprocess.run(
                [sys.executable, str(TOOLS / "run-joint-program.py"),
                 "--semantic-gemm-task-variants", "--input", str(SOURCE),
                 "--output-dir", str(output), "--amoeba-opt", str(OPT),
                 "--bm-set", "3", "--bn-set", "8", "--bk-set", "8",
                 "--k-policies", "sequential_accumulate",
                 "--reduction-topologies", "none", "--tail-policy", "reject"],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("nondivisible_static_tail", result.stderr)
            self.assertFalse(output.exists())

    def test_exact_static_tails_materialize_and_f32_reassociation_is_rejected(self):
        program = canonical_program_identity(_descriptor(5, 7, 6))
        variants = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"],
            tail_policy="exact", reduction_topologies=["balanced_tree"])
        self.assertEqual(len(variants), 1)
        with tempfile.TemporaryDirectory(prefix="orbit-semantic-tail-test-") as temporary:
            path = Path(temporary) / "tail.mlir"
            path.write_text(materialize_taskflow(program, variants[0]))
            result = subprocess.run([str(OPT), str(path), "--verify-each",
                                     f"--extract-joint-task-graph-facts=output={path}.facts.json max-tile-factor=8",
                                     "-o", "/dev/null"],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            facts = json.loads(Path(str(path) + ".facts.json").read_text())
            actual = {(edge["source_task"], edge["target_task"])
                      for edge in facts["dependencies"]}
            expected = {(edge["source_task"], edge["target_task"])
                        for edge in variants[0]["edges"]
                        if edge["kind"] != "completion_only"}
            self.assertEqual(actual, expected)
        f32 = canonical_program_identity(_descriptor(5, 7, 6, "f32"))
        f32_variants = enumerate_variants(
            f32, [4], [4], [4],
            ["sequential_accumulate", "parallel_partial_reduce"],
            tail_policy="exact")
        self.assertEqual(len(f32_variants), 1)
        self.assertEqual(f32_variants[0]["k_policy"], "sequential_accumulate")
        with tempfile.TemporaryDirectory(prefix="orbit-semantic-f32-test-") as temporary:
            path = Path(temporary) / "f32_sequential.mlir"
            path.write_text(materialize_taskflow(f32, f32_variants[0]))
            result = subprocess.run([str(OPT), str(path), "--verify-each",
                                     "-o", "/dev/null"],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_program_entry_publishes_selectable_ir_backed_graph_candidates(self):
        with tempfile.TemporaryDirectory(prefix="orbit-semantic-graph-test-") as temporary:
            output = Path(temporary)
            result = subprocess.run(
                [sys.executable, str(TOOLS / "run-joint-program.py"),
                 "--semantic-gemm-task-variants", "--input", str(SOURCE),
                 "--output-dir", str(output), "--amoeba-opt", str(OPT),
                 "--bm-set", "8", "--bn-set", "8", "--bk-set", "4",
                 "--k-policies", "sequential_accumulate,parallel_partial_reduce",
                 "--reduction-topologies", "none,linear,balanced_tree"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["semantic_variant_count"], 3)
            self.assertTrue(summary["selected_ir_verifier_pass"])
            self.assertTrue(summary["selected_cxx_facts_pass"])
            manifest = [json.loads(line) for line in
                        (output / "graph_candidate_manifest.jsonl").read_text().splitlines()]
            self.assertEqual([row["record_type"] for row in manifest],
                             ["header", "variant", "variant", "variant", "footer"])
            candidates = manifest[1:-1]
            self.assertEqual({row["action"]["reduction_topology"]
                              for row in candidates},
                             {"none", "linear", "balanced_tree"})
            self.assertEqual(len({row["graph_variant_id"] for row in candidates}), 3)
            self.assertEqual(manifest[-1]["accepted_count"], 3)
            self.assertEqual(manifest[-1]["rejection_reason_histogram"], {})
            self.assertTrue((output / "selected_taskflow.mlir").is_file())
            self.assertTrue((output / "selected_taskflow.facts.json").is_file())
            selected = json.loads((output / "taskflow_materialization_results.json").read_text())
            self.assertTrue(selected["selected"]["cxx_fact_edges_match_semantic"])
            self.assertEqual(selected["selected_graph_variant_id"],
                             summary["selected_graph_variant_id"])
            self.assertEqual(selected["source_module_sha256"],
                             manifest[0]["source_module_sha256"])


if __name__ == "__main__":
    unittest.main()
