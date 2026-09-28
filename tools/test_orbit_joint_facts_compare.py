import unittest

from orbit_joint_facts_compare import (
    GraphFactsMismatch, compare_graph_facts,
    compare_graph_facts_by_semantic_signature)


class JointFactsComparisonTest(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "tasks": [
                {"task_id": "producer", "task_role": "gemm_producer"},
                {"task_id": "consumer", "task_role": "elementwise_consumer"},
                {"task_id": "done", "task_role": "completion_join"},
            ],
            "edges": [{"source_task": "producer", "target_task": "consumer",
                       "kind": "producer_consumer", "scope": "tensor_wide",
                       "region": {"lower": [0, 0], "upper": [8, 8]}}],
        }
        self.facts = {
            "tasks": [{"task_name": "producer"}, {"task_name": "consumer"}],
            "dependencies": [{
                "source_task": "producer", "target_task": "consumer",
                "kind": "raw", "semantic_role": "producer_consumer",
                "scope": "tensor_wide", "semantic_scope": "tensor_wide",
                "region_lower": [0, 0], "region_upper": [8, 8],
            }],
            "completion_joins": [],
        }

    def test_exact_typed_region_match(self):
        self.assertTrue(compare_graph_facts(self.graph, self.facts)["pass"])

    def test_role_and_region_mutations_are_detected(self):
        self.facts["dependencies"][0]["semantic_role"] = "state_carry"
        self.assertFalse(compare_graph_facts(self.graph, self.facts)["pass"])
        self.facts["dependencies"][0]["semantic_role"] = "producer_consumer"
        self.facts["dependencies"][0]["region_upper"] = [8, 7]
        self.assertFalse(compare_graph_facts(self.graph, self.facts)["pass"])

    def test_missing_proof_is_rejected(self):
        self.facts["dependencies"][0].pop("semantic_scope")
        with self.assertRaises(GraphFactsMismatch):
            compare_graph_facts(self.graph, self.facts)

    def test_semantic_signature_maps_distinct_mlir_names(self):
        self.graph["tasks"][0].update({
            "task_role": "producer", "K_range": [0, 8],
            "original_write_memrefs": ["T"],
            "regions": {"T_output": {"lower": [0, 0], "upper": [8, 8]}}})
        self.graph["tasks"][1].update({
            "task_role": "consumer", "original_write_memrefs": ["D"],
            "regions": {"D_output": {"lower": [0, 0], "upper": [8, 8]}}})
        self.facts["tasks"] = [
            {"task_name": "producer.tile", "semantic_role": "producer",
             "semantic_k_range": [0, 8],
             "output_regions": [{"lower": [0, 0], "upper": [8, 8]}]},
            {"task_name": "consumer.tile", "semantic_role": "consumer",
             "output_regions": [{"lower": [0, 0], "upper": [8, 8]}]},
        ]
        self.facts["dependencies"][0]["source_task"] = "producer.tile"
        self.facts["dependencies"][0]["target_task"] = "consumer.tile"
        result = compare_graph_facts_by_semantic_signature(self.graph, self.facts)
        self.assertTrue(result["pass"])
        self.assertEqual(result["task_id_bijection"]["producer"],
                         "producer.tile")

    def test_semantic_signature_rejects_ambiguous_tasks(self):
        self.test_semantic_signature_maps_distinct_mlir_names()
        self.facts["tasks"].append(dict(self.facts["tasks"][0]))
        with self.assertRaises(GraphFactsMismatch):
            compare_graph_facts_by_semantic_signature(self.graph, self.facts)

    def test_typed_edges_distinguish_tree_and_final_reductions(self):
        region = {"lower": [0, 0], "upper": [8, 8]}
        graph = {"tasks": [
            {"task_id": name, "task_role": "reduction", "K_range": [0, 8],
             "original_write_memrefs": [write],
             "regions": {output: region}}
            for name, write, output in (
                ("tree", "partial_T/tree", "partial_output"),
                ("final", "T", "T_output"))],
            "edges": [{"source_task": "tree", "target_task": "final",
                       "kind": "partial_reduction", "scope": "tile_local",
                       "region": region}]}
        facts = {"tasks": [
            {"task_name": name, "semantic_role": "reduction",
             "semantic_k_range": [0, 8], "output_regions": [region]}
            for name in ("reduction.final", "reduction.tree")],
            "dependencies": [{"source_task": "reduction.tree",
                              "target_task": "reduction.final", "kind": "raw",
                              "semantic_role": "partial_reduction",
                              "scope": "tile_local",
                              "semantic_scope": "tile_local",
                              "region_lower": [0, 0],
                              "region_upper": [8, 8]}],
            "completion_joins": []}
        result = compare_graph_facts_by_semantic_signature(graph, facts)
        self.assertTrue(result["pass"])
        self.assertEqual(result["task_id_bijection"]["tree"],
                         "reduction.tree")
        facts["dependencies"].clear()
        with self.assertRaises(GraphFactsMismatch):
            compare_graph_facts_by_semantic_signature(graph, facts)


if __name__ == "__main__":
    unittest.main()
