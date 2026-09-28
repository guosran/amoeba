#!/usr/bin/env python3
"""Focused tests for the source-bound joint Taskflow reference printer.

The production rewrite/materialization path is provided by MLIR passes.  This
module is a small source-bound reference fixture generator used to check the
graph schema, exact region metadata, and verifier-facing Taskflow shape.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
REWRITE_TOOLS = Path(os.environ.get("ORBIT_REWRITE_TOOLS", str(TOOLS)))
if not REWRITE_TOOLS.is_dir():
    raise unittest.SkipTest(f"joint rewrite owner is unavailable: {REWRITE_TOOLS}")
sys.path.insert(0, str(REWRITE_TOOLS))
sys.path.insert(0, str(TOOLS))

from orbit_joint_taskflow import (  # noqa: E402
    materialize_joint_taskflow,
)


def _load_rewrite_module():
    path = REWRITE_TOOLS / "orbit_joint_rewrite_space.py"
    spec = importlib.util.spec_from_file_location(
        "orbit_joint_rewrite_space", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot load rewrite owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["orbit_joint_rewrite_space"] = module
    spec.loader.exec_module(module)
    return module


REWRITE = _load_rewrite_module()
SOURCE = (TOOLS.parent / "test/multi-cgra/taskflow/joint-scheduling/"
          "orbit_joint_pipeline_8x8_i32.mlir")
MLIR_OPT = Path(os.environ.get(
    "AMOEBA_OPT",
    "/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/mlir-amoeba-opt"))


class JointTaskflowMaterializationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file():
            raise unittest.SkipTest(f"canonical source is unavailable: {SOURCE}")
        cls.source_graph = REWRITE.root_graph(SOURCE)

    def graph(self, *actions):
        graph = self.source_graph
        for action in actions:
            result = REWRITE.apply_action(graph, action, raise_on_reject=True)
            graph = result["graph"]
        return graph

    def verify(self, text):
        if not MLIR_OPT.is_file():
            self.skipTest(f"MLIR verifier is unavailable: {MLIR_OPT}")
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as file:
            file.write(text)
            path = Path(file.name)
        try:
            completed = subprocess.run(
                [str(MLIR_OPT), "--verify-each", str(path), "-o", "/dev/null"],
                text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0,
                             msg=completed.stdout + completed.stderr)
        finally:
            path.unlink(missing_ok=True)

    def facts(self, text):
        if not MLIR_OPT.is_file():
            self.skipTest(f"MLIR facts tool is unavailable: {MLIR_OPT}")
        with tempfile.TemporaryDirectory() as directory:
            ir_path = Path(directory) / "candidate.mlir"
            facts_path = Path(directory) / "candidate.facts.json"
            ir_path.write_text(text)
            completed = subprocess.run(
                [str(MLIR_OPT), "--verify-each", str(ir_path),
                 f"--extract-joint-task-graph-facts=output={facts_path} "
                 "max-tile-factor=8", "-o", "/dev/null"],
                text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0,
                             msg=completed.stdout + completed.stderr)
            return json.loads(facts_path.read_text())

    def test_identity_and_fusion_use_source_bound_bodies(self):
        identity = materialize_joint_taskflow(self.source_graph)
        self.verify(identity)
        self.assertIn("scf.for", identity)
        self.assertIn("amoeba.semantic.incoming_edges = []", identity)
        self.assertIn("amoeba.semantic.incoming_edges = [", identity)
        self.assertEqual(identity.count("%T = memref.alloc"), 1)

        fused = materialize_joint_taskflow(
            self.graph({"kind": "FuseProducerConsumer"}))
        self.verify(fused)
        self.assertIn("task_kind = \"fused_producer_consumer\"", fused)
        self.assertNotIn("%T = memref.alloc", fused)
        self.assertEqual(fused.count("memref.load %arg3[%mi, %ni]"), 1)
        self.assertEqual(fused.count("memref.store %final"), 1)

    def test_m_and_n_tiling_have_exact_regions_and_hierarchical_joins(self):
        examples = (
            ([{"kind": "TileOutputM", "factor": 4}], 1, "axis(0)"),
            ([{"kind": "TileOutputN", "factor": 4}], 1, "axis(1)"),
            ([{"kind": "TileOutputM", "factor": 4},
              {"kind": "TileOutputN", "factor": 4}], 3, "axis(1)"),
        )
        for actions, joins, axis in examples:
            with self.subTest(actions=actions):
                text = materialize_joint_taskflow(self.graph(*actions))
                self.verify(text)
                self.assertEqual(text.count("taskflow.join states"), joins)
                self.assertIn(axis, text)
                self.assertIn("amoeba.tiling.output_axes", text)
                self.assertIn("amoeba.tiling.output_region_lower", text)
                self.assertIn("amoeba.tiling.output_region_upper", text)

    def test_tile_then_fuse_and_fuse_then_tile_eliminate_global_t(self):
        tile = ({"kind": "TileOutputM", "factor": 4},
                {"kind": "TileOutputN", "factor": 4})
        sequences = (
            tile + ({"kind": "FuseTileLocalPair"},),
            ({"kind": "FuseProducerConsumer"},) + tile,
        )
        for actions in sequences:
            with self.subTest(actions=actions):
                text = materialize_joint_taskflow(self.graph(*actions))
                self.verify(text)
                self.assertEqual(
                    text.count("task_kind = \"fused_producer_consumer\""), 4)
                self.assertNotIn("%T = memref.alloc", text)
                self.assertEqual(text.count("taskflow.join states"), 3)

    def test_sequential_and_parallel_k_paths_use_private_state(self):
        sequential = materialize_joint_taskflow(self.graph(
            {"kind": "TileReductionKSequential", "factor": 2},
            {"kind": "FuseReductionConsumer"}))
        self.verify(sequential)
        self.assertNotIn("%T = memref.alloc", sequential)
        self.assertEqual(sequential.count(" = memref.alloc() : memref<8x8xi32>"), 3)
        self.assertIn("task_kind = \"fused_producer_consumer\"", sequential)

        for topology, action in (("linear", "TileReductionKParallelLinear"),
                                 ("balanced_tree", "TileReductionKParallelTree")):
            with self.subTest(topology=topology):
                text = materialize_joint_taskflow(self.graph(
                    {"kind": action, "factor": 2},
                    {"kind": "MaterializeNumericReduction"},
                    {"kind": "FuseReductionConsumer"}))
                self.verify(text)
                self.assertNotIn("%T = memref.alloc", text)
                self.assertEqual(
                    text.count(" = memref.alloc() : memref<8x8xi32>"), 4)
                self.assertIn(
                    f'amoeba.semantic.reduction_topology = "{topology}"', text)
                self.assertIn("task_kind = \"fused_reduction_consumer\"", text)

    def test_facts_cover_edges_and_completion_joins(self):
        representatives = (
            self.graph(),
            self.graph({"kind": "TileOutputM", "factor": 4},
                       {"kind": "TileOutputN", "factor": 4}),
            self.graph({"kind": "TileReductionKParallelTree", "factor": 2},
                       {"kind": "MaterializeNumericReduction"},
                       {"kind": "FuseReductionConsumer"}),
        )
        for graph in representatives:
            with self.subTest(graph_id=graph["graph_id"]):
                text = materialize_joint_taskflow(graph)
                facts = self.facts(text)
                self.assertEqual(facts["schema"],
                                 "amoeba-joint-task-graph-facts-v1")
                self.assertTrue(facts["tasks"])
                for edge in facts["dependencies"]:
                    self.assertIn(edge["scope"], ("tensor_wide", "tile_local"))
                    self.assertTrue(edge["source_task"])
                    self.assertTrue(edge["target_task"])
                if graph is not self.source_graph:
                    self.assertTrue(facts["completion_joins"] or
                                    graph["k_policy"] == "parallel")


if __name__ == "__main__":
    unittest.main()
