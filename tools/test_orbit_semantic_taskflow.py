#!/usr/bin/env python3
"""Focused source-to-Taskflow materialization tests.

These tests exercise only semantic IR generation and the MLIR verifier.  They
never invoke a backend mapper, native cost query, replay driver, or hardware
simulator.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from orbit_semantic_taskflow import (  # noqa: E402
    materialize_taskflow,
)


def _load_variant_module():
    """Load the semantic owner from this tools directory after integration."""

    override = os.environ.get("ORBIT_VARIANT_TOOLS")
    candidates = ([Path(override)] if override else
                  [TOOLS, Path("/tmp/orbit-task-variant-semantic-worktree/tools")])
    path = next((candidate / "orbit_task_variant_space.py"
                 for candidate in candidates
                 if (candidate / "orbit_task_variant_space.py").is_file()), None)
    if path is None:
        raise unittest.SkipTest("semantic variant owner is unavailable")
    spec = importlib.util.spec_from_file_location("orbit_task_variant_space", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot load semantic variant owner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VARIANT = _load_variant_module()
CANONICAL = Path(
    "/tmp/amoeba-joint-scheduling-review/orbit_vectorcgra_memory_blocking_review/"
    "orbit_generic_blocking_mainline/gemm_8x8_i32.mlir")
MLIR_OPT = Path(os.environ.get(
    "AMOEBA_OPT",
    "/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/mlir-amoeba-opt"))


class SemanticTaskflowMaterializationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CANONICAL.is_file():
            raise unittest.SkipTest(f"canonical source is unavailable: {CANONICAL}")
        cls.program = VARIANT.canonical_program_identity(CANONICAL)

    def variant(self, bm, bn, bk, policy, topology="none"):
        values = VARIANT.enumerate_variants(
            self.program, [bm], [bn], [bk], [policy],
            reduction_topologies=[topology], tail_policy="reject")
        self.assertEqual(len(values), 1)
        return values[0]

    def verify(self, text):
        if not MLIR_OPT.is_file():
            self.skipTest(f"MLIR verifier is unavailable: {MLIR_OPT}")
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as file:
            file.write(text)
            path = Path(file.name)
        try:
            completed = subprocess.run(
                [str(MLIR_OPT), str(path), "--verify-each", "-o", "/dev/null"],
                text=True, capture_output=True, check=False)
            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr)
        finally:
            path.unlink(missing_ok=True)

    def test_seven_materialization_examples_verify(self):
        examples = (
            # M-only, N-only, and MxN output tiling.
            (4, 8, 8, "sequential_accumulate", "none"),
            (8, 4, 8, "sequential_accumulate", "none"),
            (4, 4, 8, "sequential_accumulate", "none"),
            # Sequential and both parallel reduction topologies.
            (8, 8, 4, "sequential_accumulate", "none"),
            (8, 8, 2, "parallel_partial_reduce", "linear"),
            (8, 8, 2, "parallel_partial_reduce", "balanced_tree"),
            # Full MxNxK combination with a tree reduction.
            (4, 4, 2, "parallel_partial_reduce", "balanced_tree"),
        )
        for bm, bn, bk, policy, topology in examples:
            with self.subTest(bm=bm, bn=bn, bk=bk, policy=policy,
                              topology=topology):
                variant = self.variant(bm, bn, bk, policy, topology)
                text = materialize_taskflow(self.program, variant)
                self.verify(text)
                self.assertIn(
                    f'amoeba.semantic.variant_id = "{variant["variant_id"]}"',
                    text)
                self.assertIn(
                    f'amoeba.semantic.canonical_program_id = '
                    f'"{self.program["canonical_program_id"]}"', text)

    def test_sequential_state_chain_and_single_initial_c(self):
        variant = self.variant(8, 8, 2, "sequential_accumulate")
        text = materialize_taskflow(self.program, variant)
        self.verify(text)
        self.assertEqual(text.count('amoeba.semantic.initial_C_ownership = '
                                  '"external_initial_C"'), 1)
        self.assertEqual(text.count('amoeba.semantic.final_C_ownership = '
                                  '"external_final_C"'), 1)
        self.assertIn(
            "will_reads(%A, %B, %gemm_tile_m0_n0_k_0_0_2",
            text)
        self.assertIn(
            "will_writes(%alloc_state_C_gemm_tile_m0_n0_k_0_0_2",
            text)
        self.assertIn(
            "will_reads(%A, %B, %gemm_tile_m0_n0_k_0_0_2",
            text)
        self.assertIn("will_writes(%C : memref<8x8xi32>)", text)

    def test_parallel_linear_has_numeric_reduction_and_private_partials(self):
        variant = self.variant(8, 8, 2, "parallel_partial_reduce", "linear")
        text = materialize_taskflow(self.program, variant)
        self.verify(text)
        self.assertIn('amoeba.semantic.task_role = "reduction"', text)
        self.assertIn("reduce_linear_tile_m0_n0_step0", text)
        self.assertIn("amoeba.semantic.partial_C_count = 2 : i64", text)
        self.assertIn("will_writes(%alloc_partial_C_", text)
        self.assertNotIn(
            "taskflow.task @gemm_tile_m0_n0_k_0_0_2\n"
            "        will_reads(%A, %B, %C", text)

    def test_parallel_tree_has_binary_reduction_nodes(self):
        variant = self.variant(8, 8, 2, "parallel_partial_reduce", "balanced_tree")
        text = materialize_taskflow(self.program, variant)
        self.verify(text)
        self.assertIn("reduce_tree_tile_m0_n0_l0_n0", text)
        self.assertIn("reduce_tree_tile_m0_n0_l1_n0", text)
        self.assertIn('amoeba.semantic.reduction_topology = "balanced_tree"', text)
        self.assertGreaterEqual(text.count('amoeba.semantic.task_role = "reduction"'), 3)

    def test_mn_tiles_use_hierarchical_completion_only_joins(self):
        variant = self.variant(4, 4, 8, "sequential_accumulate")
        text = materialize_taskflow(self.program, variant)
        self.verify(text)
        # Two row joins along N and one whole-result join along M.
        self.assertEqual(text.count("taskflow.join states"), 3)
        self.assertIn("axis(1) region([0, 0], [4, 8])", text)
        self.assertIn("axis(0) region([0, 0], [8, 8])", text)
        self.assertNotIn("task_role = \"reduction\"", text)

    def test_mnk_facts_use_per_operand_regions_for_disjoint_tiles(self):
        """C++ facts must retain only the eight semantic partial edges."""

        variant = self.variant(4, 4, 4, "parallel_partial_reduce", "linear")
        text = materialize_taskflow(self.program, variant)
        self.verify(text)
        self.assertIn("amoeba.tiling.input_region_lowers", text)
        self.assertIn("amoeba.tiling.input_region_uppers", text)
        self.assertIn("amoeba.tiling.input_region_reasons", text)
        self.assertIn("amoeba.tiling.output_region_lowers", text)
        self.assertIn("amoeba.tiling.output_region_uppers", text)
        if not MLIR_OPT.is_file():
            self.skipTest(f"MLIR facts tool is unavailable: {MLIR_OPT}")
        with tempfile.TemporaryDirectory() as directory:
            ir_path = Path(directory) / "mnk.mlir"
            facts_path = Path(directory) / "mnk.facts.json"
            ir_path.write_text(text)
            completed = subprocess.run(
                [str(MLIR_OPT), str(ir_path), "--verify-each",
                 f"--extract-joint-task-graph-facts=output={facts_path} "
                 "max-tile-factor=8", "-o", "/dev/null"],
                text=True, capture_output=True, check=False)
            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr)
            facts = json.loads(facts_path.read_text())
        edges = facts["dependencies"]
        direct = [edge for edge in edges
                  if edge["source_task"].startswith("gemm_") and
                  edge["target_task"].startswith("reduce_")]
        self.assertEqual(len(direct), 8)
        memory_order = [edge for edge in edges
                        if edge.get("origin") == "memory_order"]

        def tile(task_name):
            match = re.search(r"tile_m\d+_n\d+", task_name)
            return match.group(0) if match else None

        self.assertEqual(
            [(edge["source_task"], edge["target_task"])
             for edge in memory_order
             if tile(edge["source_task"]) != tile(edge["target_task"])], [])


if __name__ == "__main__":
    unittest.main()
