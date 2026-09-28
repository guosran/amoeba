#!/usr/bin/env python3
"""Tests for strict whole-program ranking across graph variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from joint_program_ranking import (
    CANDIDATE_SCHEMA,
    COST_SCHEMA,
    GRAPH_SCHEMA,
    SCORE_SCHEMA,
    RankingError,
    VariantArtifacts,
    build_global_ranking,
    load_graph_variants,
    sha256_file,
    write_global_ranking,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JointProgramRankingTest(unittest.TestCase):
    def _compact_scores(self, artifacts: VariantArtifacts, top_k: int) -> None:
        records = [
            json.loads(line)
            for line in artifacts.scores.read_text(encoding="utf-8").splitlines()
        ]
        scores = sorted(
            records[1:-1],
            key=lambda score: (
                score["predicted_scheduler_makespan"],
                int(score["candidate_id"].split("-", 1)[1]),
            ),
        )
        retained = scores[:top_k]
        records[0]["output_scope"] = "selected-top-k"
        records[-1].update(
            {
                "evaluated_count": len(scores),
                "evaluated_valid_count": len(scores),
                "scored_count": len(retained),
                "valid_count": len(retained),
                "top_k_requested": top_k,
                "shortlist": [
                    {
                        "rank": rank,
                        "candidate_id": score["candidate_id"],
                        "predicted_scheduler_makespan": score[
                            "predicted_scheduler_makespan"
                        ],
                    }
                    for rank, score in enumerate(retained)
                ],
            }
        )
        _write_jsonl(artifacts.scores, [records[0], *retained, records[-1]])

    def _space(self, root: Path) -> tuple[Path, Path, list]:
        source = root / "input.mlir"
        source.write_text("module {}\n", encoding="utf-8")
        variants_dir = root / "variants"
        variants_dir.mkdir()
        identifiers = ("graph-000000", "graph-000003")
        variant_records = []
        for index, identifier in enumerate(identifiers):
            ir = variants_dir / f"{identifier}.mlir"
            ir.write_text(
                source.read_text(encoding="utf-8")
                if index == 0
                else f"module {{ // {identifier}\n}}\n",
                encoding="utf-8",
            )
            actions = (
                []
                if index == 0
                else [
                    {
                        "kind": "tiling",
                        "task_name": "A",
                        "axis": 0,
                        "factor": 2,
                    }
                ]
            )
            variant_records.append(
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "variant",
                    "graph_variant_id": identifier,
                    "graph_variant_sha256": sha256_file(ir),
                    "lineage_path_id": f"path-{index}",
                    "parent_graph_variant_id": None if index == 0 else identifiers[0],
                    "depth": index,
                    "actions": actions,
                    "structural_sha256": f"{index + 1:064x}",
                    "result_ir": {
                        "path": ir.name,
                        "path_base": "artifact_dir",
                        "sha256": sha256_file(ir),
                    },
                    "semantic_witness": (
                        {
                            "status": "verified",
                            "scope": "byte-identical-source",
                            "source_sha256": sha256_file(source),
                            "candidate_sha256": sha256_file(ir),
                        }
                        if index == 0
                        else {
                            "status": "verified",
                            "command_template": "witness {source} {candidate}",
                            "graph_variant_id": identifier,
                            "source_sha256": sha256_file(source),
                            "candidate_sha256": sha256_file(ir),
                            "actions_sha256": _canonical_sha256(actions),
                        }
                    ),
                }
            )
        graph = root / "graphs.jsonl"
        _write_jsonl(
            graph,
            [
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "header",
                    "source": {"path": str(source), "sha256": sha256_file(source)},
                    "artifact_dir": {
                        "path": "variants",
                        "path_base": "manifest_dir",
                    },
                    "preparation": {
                        "contract": "amoeba-graph-variant-preparation-v1",
                    },
                },
                variant_records[0],
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "duplicate",
                    "duplicates_graph_variant_id": identifiers[0],
                },
                variant_records[1],
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "traversal",
                    "path_id": "path-0",
                    "parent_path_id": None,
                    "graph_variant_id": identifiers[0],
                    "parent_graph_variant_id": None,
                    "depth": 0,
                    "actions": [],
                    "structural_sha256": f"{1:064x}",
                    "published": True,
                },
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "traversal",
                    "path_id": "path-1",
                    "parent_path_id": "path-0",
                    "graph_variant_id": identifiers[1],
                    "parent_graph_variant_id": identifiers[0],
                    "depth": 1,
                    "actions": variant_records[1]["actions"],
                    "structural_sha256": f"{2:064x}",
                    "published": True,
                },
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "traversal",
                    "path_id": "path-2",
                    "parent_path_id": "path-1",
                    "graph_variant_id": identifiers[1],
                    "parent_graph_variant_id": identifiers[1],
                    "depth": 2,
                    "actions": [
                        *variant_records[1]["actions"],
                        {
                            "kind": "tiling",
                            "task_name": "A.tile.0",
                            "axis": 1,
                            "factor": 2,
                        },
                    ],
                    "structural_sha256": f"{2:064x}",
                    "published": False,
                },
                {
                    "schema": GRAPH_SCHEMA,
                    "record_type": "footer",
                    "complete": True,
                    "unique_graph_variants": 2,
                    "traversal_states": 3,
                    "duplicate_rewrite_sequences": 1,
                },
            ],
        )
        architecture = root / "architecture.yaml"
        architecture.write_text("architecture: test\n", encoding="utf-8")
        return graph, architecture, load_graph_variants(graph)[1]

    def _variant_artifacts(
        self,
        root: Path,
        architecture: Path,
        graph,
        makespans: list[int],
    ) -> VariantArtifacts:
        directory = root / graph.graph_variant_id
        directory.mkdir()
        candidates = directory / "candidates.jsonl"
        candidate_ids = [f"candidate-{index}" for index in range(len(makespans))]
        _write_jsonl(
            candidates,
            [
                {
                    "schema": CANDIDATE_SCHEMA,
                    "record_type": "header",
                    "function": "main",
                    "graph_variant_id": graph.graph_variant_id,
                    "graph_variant_sha256": graph.ir_sha256,
                    "architecture": {"spec_sha256": sha256_file(architecture)},
                },
                *[
                    {
                        "schema": CANDIDATE_SCHEMA,
                        "record_type": "candidate",
                        "candidate_id": candidate_id,
                    }
                    for candidate_id in candidate_ids
                ],
                {
                    "schema": CANDIDATE_SCHEMA,
                    "record_type": "footer",
                    "candidate_count": len(candidate_ids),
                },
            ],
        )
        costs = directory / "costs.json"
        costs.write_text(
            json.dumps({"schema": COST_SCHEMA, "namespace": "fixture"}) + "\n",
            encoding="utf-8",
        )
        scores = directory / "scores.jsonl"
        _write_jsonl(
            scores,
            [
                {
                    "schema": SCORE_SCHEMA,
                    "record_type": "header",
                    "candidate_schema": CANDIDATE_SCHEMA,
                    "candidate_manifest_sha256": sha256_file(candidates),
                    "cost_catalog_sha256": sha256_file(costs),
                    "cost_namespace": "fixture",
                    "architecture_sha256": sha256_file(architecture),
                    "communication_mode": "explicit",
                    "schedule_trace_schema": "joint-schedule-trace-v3",
                    "placement_search_policy":
                        "global-fixed-order-exact-v2",
                    "activity_scheduling_policy":
                        "fixed-task-priority-v1",
                    "activity_beam_width": 1,
                    "function": "main",
                    "graph_variant_id": graph.graph_variant_id,
                    "graph_variant_sha256": graph.ir_sha256,
                },
                *[
                    {
                        "schema": SCORE_SCHEMA,
                        "record_type": "score",
                        "candidate_id": candidate_id,
                        "valid": True,
                        "predicted_scheduler_makespan": makespan,
                        "schedule_trace_sha256": f"{makespan:064x}",
                    }
                    for candidate_id, makespan in zip(candidate_ids, makespans)
                ],
                {
                    "schema": SCORE_SCHEMA,
                    "record_type": "footer",
                    "candidate_count": len(candidate_ids),
                    "scored_count": len(candidate_ids),
                    "valid_count": len(candidate_ids),
                },
            ],
        )
        return VariantArtifacts(graph, candidates, costs, scores)

    def test_global_order_uses_manifest_indices_not_graph_id_text(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-") as raw:
            root = Path(raw)
            graph_manifest, architecture, graphs = self._space(root)
            artifacts = [
                self._variant_artifacts(root, architecture, graphs[0], [12, 8]),
                self._variant_artifacts(root, architecture, graphs[1], [8]),
            ]
            ranking = build_global_ranking(
                graph_manifest, architecture, artifacts, "explicit", 2, 3
            )
            self.assertEqual(ranking["counts"]["total_candidates"], 3)
            self.assertEqual(
                [
                    (row["graph_variant_id"], row["candidate_id"])
                    for row in ranking["shortlist"]
                ],
                [("graph-000000", "candidate-1"), ("graph-000003", "candidate-0")],
            )
            output = root / "global-ranking.json"
            write_global_ranking(output, ranking)
            self.assertTrue(json.loads(output.read_text())["complete"])

    def test_global_ranking_accepts_an_explicit_manifest_ordered_subset(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-subset-") as raw:
            root = Path(raw)
            graph_manifest, architecture, graphs = self._space(root)
            artifact = self._variant_artifacts(
                root, architecture, graphs[1], [8]
            )
            ranking = build_global_ranking(
                graph_manifest,
                architecture,
                [artifact],
                "explicit",
                1,
                1,
                selected_graph_variant_ids=[graphs[1].graph_variant_id],
            )
            self.assertEqual(ranking["counts"]["graph_variants"], 1)
            self.assertEqual(
                ranking["selected_graph_variant_ids"],
                [graphs[1].graph_variant_id],
            )

            with self.assertRaisesRegex(RankingError, "absent"):
                build_global_ranking(
                    graph_manifest,
                    architecture,
                    [artifact],
                    "explicit",
                    1,
                    1,
                    selected_graph_variant_ids=["graph-999999"],
                )

    def test_compact_local_top_k_preserves_global_top_k(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-compact-") as raw:
            root = Path(raw)
            graph_manifest, architecture, graphs = self._space(root)
            artifacts = [
                self._variant_artifacts(
                    root, architecture, graphs[0], [12, 8, 20]
                ),
                self._variant_artifacts(
                    root, architecture, graphs[1], [9, 7, 30]
                ),
            ]
            for item in artifacts:
                self._compact_scores(item, 2)
            ranking = build_global_ranking(
                graph_manifest, architecture, artifacts, "explicit", 2, 6
            )
            self.assertEqual(ranking["counts"]["total_candidates"], 6)
            self.assertEqual(ranking["counts"]["valid_candidates"], 6)
            self.assertEqual(
                [
                    (row["graph_variant_id"], row["candidate_id"])
                    for row in ranking["shortlist"]
                ],
                [("graph-000003", "candidate-1"),
                 ("graph-000000", "candidate-1")],
            )
            with self.assertRaisesRegex(RankingError, "footer counts"):
                build_global_ranking(
                    graph_manifest, architecture, artifacts, "explicit", 3, 6
                )

    def test_tampered_graph_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-tamper-") as raw:
            root = Path(raw)
            graph_manifest, _, graphs = self._space(root)
            graphs[1].ir_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RankingError, "result IR mismatch"):
                load_graph_variants(graph_manifest)

    def test_identity_variant_must_be_byte_identical_to_source(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-identity-") as raw:
            root = Path(raw)
            graph_manifest, _, graphs = self._space(root)
            identity = graphs[0].ir_path
            identity.write_text("module { // not identity\n}\n", encoding="utf-8")
            records = [
                json.loads(line)
                for line in graph_manifest.read_text(encoding="utf-8").splitlines()
            ]
            identity_sha = sha256_file(identity)
            records[1]["graph_variant_sha256"] = identity_sha
            records[1]["result_ir"]["sha256"] = identity_sha
            records[1]["semantic_witness"]["candidate_sha256"] = identity_sha
            _write_jsonl(graph_manifest, records)
            with self.assertRaisesRegex(RankingError, "invalid identity witness"):
                load_graph_variants(graph_manifest)

    def test_rewrite_witness_candidate_hash_must_match(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-witness-ir-") as raw:
            root = Path(raw)
            graph_manifest, _, _ = self._space(root)
            records = [
                json.loads(line)
                for line in graph_manifest.read_text(encoding="utf-8").splitlines()
            ]
            records[3]["semantic_witness"]["candidate_sha256"] = "0" * 64
            _write_jsonl(graph_manifest, records)
            with self.assertRaisesRegex(RankingError, "byte identity mismatch"):
                load_graph_variants(graph_manifest)

    def test_rewrite_witness_actions_must_match(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-witness-actions-") as raw:
            root = Path(raw)
            graph_manifest, _, _ = self._space(root)
            records = [
                json.loads(line)
                for line in graph_manifest.read_text(encoding="utf-8").splitlines()
            ]
            records[3]["actions"][0]["factor"] = 4
            records[5]["actions"][0]["factor"] = 4
            records[6]["actions"][0]["factor"] = 4
            _write_jsonl(graph_manifest, records)
            with self.assertRaisesRegex(RankingError, "witness action mismatch"):
                load_graph_variants(graph_manifest)

    def test_traversal_lineage_must_extend_parent(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-lineage-") as raw:
            root = Path(raw)
            graph_manifest, _, _ = self._space(root)
            records = [
                json.loads(line)
                for line in graph_manifest.read_text(encoding="utf-8").splitlines()
            ]
            records[6]["actions"][0]["factor"] = 4
            _write_jsonl(graph_manifest, records)
            with self.assertRaisesRegex(RankingError, "does not extend its parent"):
                load_graph_variants(graph_manifest)

    def test_total_budget_and_late_variant_failure_publish_nothing(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-atomic-") as raw:
            root = Path(raw)
            graph_manifest, architecture, graphs = self._space(root)
            artifacts = [
                self._variant_artifacts(root, architecture, graphs[0], [7, 9]),
                self._variant_artifacts(root, architecture, graphs[1], [8]),
            ]
            output = root / "global-ranking.json"
            with self.assertRaisesRegex(RankingError, "total candidate budget"):
                ranking = build_global_ranking(
                    graph_manifest, architecture, artifacts, "explicit", 1, 2
                )
                write_global_ranking(output, ranking)
            self.assertFalse(output.exists())

            artifacts[1].scores.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(RankingError):
                ranking = build_global_ranking(
                    graph_manifest, architecture, artifacts, "explicit", 1, 3
                )
                write_global_ranking(output, ranking)
            self.assertFalse(output.exists())

    def test_graph_provenance_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="joint-ranking-graph-id-") as raw:
            root = Path(raw)
            graph_manifest, architecture, graphs = self._space(root)
            artifacts = [
                self._variant_artifacts(root, architecture, graphs[0], [7]),
                self._variant_artifacts(root, architecture, graphs[1], [8]),
            ]
            records = [
                json.loads(line)
                for line in artifacts[1].scores.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["graph_variant_id"] = graphs[0].graph_variant_id
            _write_jsonl(artifacts[1].scores, records)
            with self.assertRaisesRegex(RankingError, "provenance"):
                build_global_ranking(
                    graph_manifest, architecture, artifacts, "explicit", 1, 2
                )


if __name__ == "__main__":
    raise SystemExit(unittest.main())
