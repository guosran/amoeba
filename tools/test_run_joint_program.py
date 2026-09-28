#!/usr/bin/env python3
"""Focused fail-closed tests for prescored ORBIT resume provenance."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "orbit_run_joint_program", TOOLS / "run-joint-program.py"
)
assert spec is not None and spec.loader is not None
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


class ResumeContractTest(unittest.TestCase):
    def test_explicit_predictor_root_is_provenance_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictor_root = root / "candidate-predictor"
            adapters = predictor_root / "adapters"
            adapters.mkdir(parents=True)
            for name in (
                "extract_amoeba_task_dfgs.py",
                "generate_amoeba_query_features.py",
                "amoeba_reference_cost_catalog.py",
            ):
                (adapters / name).write_text("# candidate adapter\n")
            checkpoint = predictor_root / "checkpoint.json"
            checkpoint.write_text(json.dumps({
                "schema": "cgra-ii-reference-checkpoint-v1",
            }))
            args = argparse.Namespace(
                predictor_checkpoint=checkpoint,
                predictor_root=predictor_root,
                predictor_catalog_mode="auto",
                allow_provisional_checkpoint=False,
                allow_test_fixture=False,
                max_cgras_per_task=4,
            )
            inputs = driver._required_predictor_inputs(args)
            self.assertEqual(inputs["root"], predictor_root.resolve())
            self.assertEqual(inputs["checkpoint"], checkpoint.resolve())
            self.assertEqual(inputs["catalog_mode"], "reference")
            self.assertEqual(
                inputs["catalog"],
                adapters.resolve() / "amoeba_reference_cost_catalog.py",
            )

            args.predictor_root = root / "missing"
            with self.assertRaisesRegex(
                    driver.PipelineError, "predictor root is not a directory"):
                driver._required_predictor_inputs(args)

    def test_formal_ensemble_resolves_sealed_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapters = root / "adapters"
            model_dir = root / "models" / "formal"
            adapters.mkdir()
            model_dir.mkdir(parents=True)
            for name in (
                "extract_amoeba_task_dfgs.py",
                "generate_amoeba_query_features.py",
                "amoeba_cost_catalog.py",
            ):
                (adapters / name).write_text("# formal adapter\n")
            checkpoint_records = {}
            for name in ("baseline", "large-operation", "ranking"):
                path = model_dir / f"{name}.pt"
                path.write_bytes(name.encode())
                checkpoint_records[name] = {"path": path.name}
            ensemble = model_dir / "ensemble.json"
            ensemble.write_text(json.dumps({
                "schema": driver.VARIANT_DRIVER.FORMAL_ENSEMBLE_SCHEMA,
                "checkpoints": checkpoint_records,
            }))
            args = argparse.Namespace(
                predictor_checkpoint=ensemble,
                predictor_root=root,
                predictor_catalog_mode="auto",
                allow_provisional_checkpoint=False,
                allow_test_fixture=False,
                max_cgras_per_task=4,
            )
            inputs = driver._required_predictor_inputs(args)
            self.assertEqual(inputs["catalog_mode"], "formal-ensemble")
            self.assertEqual(
                inputs["catalog"], adapters / "amoeba_cost_catalog.py"
            )
            self.assertEqual(
                set(inputs["checkpoint_paths"]),
                {"baseline", "large-operation", "ranking"},
            )

            args.allow_provisional_checkpoint = True
            with self.assertRaisesRegex(
                    driver.PipelineError, "forbids provisional"):
                driver._required_predictor_inputs(args)
            args.allow_provisional_checkpoint = False
            args.max_cgras_per_task = 0
            with self.assertRaisesRegex(
                    driver.PipelineError, "requires --max-cgras-per-task"):
                driver._required_predictor_inputs(args)

    def test_explicit_replay_selection_is_strict(self) -> None:
        self.assertEqual(
            driver._parse_replay_selection("graph-000100:candidate-3"),
            ("graph-000100", "candidate-3"),
        )
        for malformed in (
            "graph-000100", "graph-x:candidate-3", "graph-1:candidate-x",
            "graph-1:candidate-2:extra",
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(argparse.ArgumentTypeError):
                    driver._parse_replay_selection(malformed)

    def test_manifest_function_must_match_explicit_request(self) -> None:
        header = {"function": "main"}
        self.assertEqual(
            driver._validated_manifest_function("", header), "main"
        )
        self.assertEqual(
            driver._validated_manifest_function("main", header), "main"
        )
        with self.assertRaisesRegex(
            driver.PipelineError, "does not match requested function"
        ):
            driver._validated_manifest_function("other", header)
        with self.assertRaisesRegex(
            driver.PipelineError, "manifest function is invalid"
        ):
            driver._validated_manifest_function("", {"function": ""})

    def test_resume_output_cannot_mutate_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mlir"
            architecture = root / "architecture.yaml"
            optimizer = root / "mlir-amoeba-opt"
            resume = root / "resume"
            source.write_text("module {}\n", encoding="utf-8")
            architecture.write_text("grid: 4x4\n", encoding="utf-8")
            optimizer.write_text("#!/bin/sh\n", encoding="utf-8")
            optimizer.chmod(0o755)
            resume.mkdir()
            marker = resume / "marker"
            marker.write_text("unchanged\n", encoding="utf-8")
            before = sorted(
                (path.relative_to(resume), path.read_bytes())
                for path in resume.rglob("*") if path.is_file()
            )
            args = argparse.Namespace(
                input=source,
                architecture=architecture,
                amoeba_opt=optimizer,
                resume_prescored_dir=resume,
                output_dir=resume / "new" / "output",
            )
            with mock.patch.object(
                    driver, "_required_predictor_inputs", return_value={}):
                with self.assertRaisesRegex(
                        driver.PipelineError, "must not be inside"):
                    driver.run(args)
            after = sorted(
                (path.relative_to(resume), path.read_bytes())
                for path in resume.rglob("*") if path.is_file()
            )
            self.assertEqual(before, after)
            self.assertFalse((resume / "new").exists())

            # A failure after the sibling transaction has been created and
            # populated must also leave the source byte-for-byte unchanged
            # and remove the disposable copy.
            (resume / "graph-variants.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            before = sorted(
                (path.relative_to(resume), path.read_bytes())
                for path in resume.rglob("*") if path.is_file()
            )
            args.output_dir = root / "output"
            args.graph_manifest_input = None
            with mock.patch.object(
                    driver, "_required_predictor_inputs", return_value={}):
                with self.assertRaises(driver.RankingError):
                    driver.run(args)
            after = sorted(
                (path.relative_to(resume), path.read_bytes())
                for path in resume.rglob("*") if path.is_file()
            )
            self.assertEqual(before, after)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(
                [path for path in root.iterdir()
                 if path.name.startswith(".output.")],
                [],
            )

    def test_checkpoint_and_search_contract_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            architecture = root / "architecture.yaml"
            optimizer = root / "mlir-amoeba-opt"
            checkpoint = root / "checkpoint.pt"
            architecture.write_text("grid: 4x4\n", encoding="utf-8")
            optimizer.write_bytes(b"optimizer")
            checkpoint.write_bytes(b"checkpoint-v1")
            architecture_sha = driver.sha256_file(architecture)
            optimizer_sha = driver.sha256_file(optimizer)
            checkpoint_sha = driver.sha256_file(checkpoint)
            body_sha = "b" * 64
            task_dfg = root / "task.mlir"
            task_dfg.write_text("module {}\n", encoding="utf-8")
            task_dfg_sha = driver.sha256_file(task_dfg)
            task_index = root / "task-index.json"
            task_index.write_text(
                json.dumps({"A": task_dfg.name}), encoding="utf-8"
            )

            candidates = root / "candidates.jsonl"
            header = {
                "record_type": "header",
                "schema": "amoeba-analytical-task-candidates",
                "resource_axis_version": 3,
                "search_scope": "static-shape-projected-sample",
                "max_cgras_per_task": 16,
                "max_replicas_per_task": 4,
                "max_replica_placement_alternatives": 0,
                "sample_candidates": 20000,
                "sample_seed": 0,
                "candidate_budget": {"max_candidates": 1000000},
                "architecture": {
                    "grid_rows": 4,
                    "grid_cols": 4,
                    "spec_sha256": architecture_sha,
                },
                "tasks": [{"task": "A", "body_sha256": body_sha}],
            }
            candidates.write_text(
                json.dumps(header) + "\n" + json.dumps({
                    "record_type": "footer",
                    "schema": "amoeba-analytical-task-candidates",
                    "candidate_count": 0,
                }) + "\n",
                encoding="utf-8",
            )
            applicability = {
                "workload_domain": "static_tensor",
                "static_shapes": True,
                "declaration_source": "validated_task_dfg_analysis",
                "validated_task_count": 1,
            }
            input_provenance = {
                "architecture_sha256": architecture_sha,
                "neura_opt_sha256": optimizer_sha,
                "task_body_sha256": {"A": body_sha},
                "task_dfg_sha256": {"A": task_dfg_sha},
                "rec_res_source": "analysis",
                "startup_cycles_source": "analysis",
                "applicability": applicability,
            }
            predictor_input = root / "input.json"
            predictor_input.write_text(json.dumps({
                "provenance": input_provenance,
            }), encoding="utf-8")
            fabric = {"contract_id": "test-fabric"}
            training_evidence = {"source": "test"}
            current_model = {
                "feature_names_sha256": "f" * 64,
                "feature_contract_id": "test-features",
                "feature_extractor_source_sha256": "e" * 64,
                "fabric": fabric,
                "applicability": {"workload_domains": ["static_tensor"]},
                "parameter_count": 12161,
                "quality_status": "production",
                "training_evidence": training_evidence,
            }
            config_sha = driver.hashlib.sha256(json.dumps({
                "feature_names_sha256": current_model[
                    "feature_names_sha256"
                ],
                "feature_contract_id": current_model["feature_contract_id"],
                "feature_extractor_source_sha256": current_model[
                    "feature_extractor_source_sha256"
                ],
                "fabric": fabric,
                "applicability": current_model["applicability"],
                "parameter_count": current_model["parameter_count"],
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            costs = root / "costs.json"
            costs.write_text(json.dumps({
                "predictor_metadata": {
                    "candidate_manifest_sha256":
                        driver.sha256_file(candidates),
                    "analytical_input_sha256":
                        driver.sha256_file(predictor_input),
                    "model": {
                        "sha256": checkpoint_sha,
                        "config_sha256": config_sha,
                        "quality_status": "production",
                        "parameter_count": 12161,
                        "feature_contract_id": "test-features",
                        "feature_extractor_source_sha256": "e" * 64,
                        "training_evidence": training_evidence,
                    },
                    "architecture_contract": {
                        "contract_id": "test-fabric",
                        "fabric": fabric,
                        "shape_protocol": "test-shapes",
                        "supported_architecture_sha256": [architecture_sha],
                        "architecture_sha256_role":
                            "catalog_provenance_for_validated_fabric",
                    },
                    "analytical_provenance": input_provenance,
                },
            }), encoding="utf-8")
            graph = driver.GraphVariant(
                0, "graph-0", root / "graph.mlir", "0" * 64, {}
            )
            run = driver.VariantRun(
                graph, root, root / "prepared.mlir", candidates,
                root / "bound.mlir", root / "route.mlir",
                task_index, predictor_input, costs,
                root / "scores.jsonl", root / "scored.mlir",
            )
            args = argparse.Namespace(
                max_graph_depth=3, max_graph_variants=1000,
                min_tile_factor=2, max_tile_factor=8, tile_factors=None,
                forbid_nested_tiling=False,
                max_fission_actions_per_task=256,
                max_duplicate_samples=100, max_fusion_steps=2,
                max_fission_steps=0, max_tiling_steps=2,
                allow_test_fixture=False, allow_provisional_checkpoint=False,
                max_cgras_per_task=0, max_replicas_per_task=4,
                max_replica_placement_alternatives=0,
                shape_search_policy="projected-sample",
                max_changed_tasks=2, sample_candidates=20000,
                sample_seed=0, max_candidates_per_variant=1000000,
                workload_domain="static_tensor",
            )
            graph_header = {"bounds": {
                "max_depth": 3, "max_variants": 1000,
                "min_tile_factor": 2, "max_tile_factor": 8,
                "tile_factors": [1, 2, 4, 8],
                "tile_factor_one_is_identity": True,
                "forbid_nested_tiling": False,
                "max_fission_actions_per_task": 256,
                "fission_fact_generation_enabled": False,
                "max_duplicate_samples": 100,
                "max_fusion_steps": 2, "max_fission_steps": 0,
                "max_tiling_steps": 2,
            }}
            fake_adapter = types.SimpleNamespace(
                load_reference_checkpoint=lambda *args, **kwargs:
                    (None, current_model),
                source_task_body_sha256=lambda text, task: (
                    body_sha if text == "module {}\n" else "c" * 64
                ),
                SHAPE_PROTOCOL_ID="test-shapes",
            )
            predictor = {"checkpoint": checkpoint, "catalog": root / "adapter.py"}
            with mock.patch.object(
                    driver, "_load_catalog_driver",
                    return_value=fake_adapter):
                contract = driver._validate_resume_contract(
                    args, architecture, optimizer, predictor, graph_header, [run]
                )
                self.assertEqual(contract["resource_axis_version"], 3)

                original_costs = costs.read_text(encoding="utf-8")
                changed_costs = json.loads(original_costs)
                changed_costs["predictor_metadata"]["model"][
                    "feature_contract_id"
                ] = "stale-features"
                costs.write_text(json.dumps(changed_costs), encoding="utf-8")
                with self.assertRaisesRegex(
                        driver.PipelineError, "predictor provenance mismatch"):
                    driver._validate_resume_contract(
                        args, architecture, optimizer, predictor,
                        graph_header, [run]
                    )
                costs.write_text(original_costs, encoding="utf-8")

                task_dfg.write_text("module { changed }\n", encoding="utf-8")
                with self.assertRaisesRegex(
                        driver.PipelineError, "task body provenance mismatch"):
                    driver._validate_resume_contract(
                        args, architecture, optimizer, predictor,
                        graph_header, [run]
                    )
                task_dfg.write_text("module {}\n", encoding="utf-8")

                checkpoint.write_bytes(b"checkpoint-v2")
                with self.assertRaisesRegex(
                        driver.PipelineError, "predictor provenance mismatch"):
                    driver._validate_resume_contract(
                        args, architecture, optimizer, predictor,
                        graph_header, [run]
                    )
                checkpoint.write_bytes(b"checkpoint-v1")
                predictor_input.write_text(
                    json.dumps({
                        "provenance": input_provenance,
                        "changed": True,
                    }),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                        driver.PipelineError, "predictor provenance mismatch"):
                    driver._validate_resume_contract(
                        args, architecture, optimizer, predictor,
                        graph_header, [run]
                    )


if __name__ == "__main__":
    unittest.main()
