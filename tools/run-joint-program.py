#!/usr/bin/env python3
"""Run transformation-aware joint scheduling as one atomic program search.

The pipeline enumerates witnessed task-graph variants first, prepares every
variant under one versioned lowering contract, freezes its declared task-shape
space, and checks the global candidate budget before invoking the predictor.
It then scores every declared candidate, ranks the union, and runs the real
per-task mapper plus resource binder for an explicitly bounded global top-k
or an explicit provenance-bound graph/candidate ablation selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, List, Mapping, Optional, Sequence

import joint_graph_variant_space
from joint_graph_variant_space import (
    PREPARATION_PASSES,
    GraphSpaceError,
    enumerate_graph_space,
)
from joint_program_ranking import (
    GraphVariant,
    RankingError,
    VariantArtifacts,
    build_global_ranking,
    load_graph_variants,
    sha256_file,
    write_global_ranking,
)


TOOLS_ROOT = Path(__file__).resolve().parent
AMOEBA_ROOT = TOOLS_ROOT.parent
PREDICTOR_ROOT = AMOEBA_ROOT / "thirdparty" / "cgra-ii-predictor"
ADAPTER_ROOT = PREDICTOR_ROOT / "adapters"


def _load_variant_driver() -> Any:
    path = TOOLS_ROOT / "run-joint-scheduling.py"
    spec = importlib.util.spec_from_file_location("amoeba_joint_variant", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load per-variant pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VARIANT_DRIVER = _load_variant_driver()
PipelineError = VARIANT_DRIVER.PipelineError


_REPLAY_SELECTION = re.compile(r"^(graph-[0-9]+):(candidate-[0-9]+)$")


def _parse_replay_selection(raw: str) -> tuple[str, str]:
    match = _REPLAY_SELECTION.fullmatch(raw)
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected GRAPH_VARIANT_ID:CANDIDATE_ID"
        )
    return match.group(1), match.group(2)


def _load_catalog_driver(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "amoeba_shared_cost_catalog", path
    )
    if spec is None or spec.loader is None:
        raise PipelineError(f"cannot load cost-catalog adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class VariantRun:
    graph: GraphVariant
    root: Path
    prepared: Path
    candidates: Path
    bound: Path
    route_expanded: Path
    task_index: Path
    predictor_input: Path
    costs: Path
    scores: Path
    scored: Path
    candidate_count: int = 0


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _new_output_transaction(final: Path) -> tuple[Path, Path]:
    final = final.expanduser().resolve()
    if final.exists():
        raise PipelineError(f"output directory already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    return final, stage


def _copy_graph_manifest_with_rebased_artifacts(
    supplied: Path, output: Path, artifact_directory: str, source: Path,
) -> None:
    """Copy a graph manifest while rebasing its bundle-local paths."""
    with supplied.open("r", encoding="utf-8") as input_stream:
        first_line = input_stream.readline()
        try:
            header = json.loads(first_line)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"invalid pre-enumerated graph manifest header: {error}"
            ) from error
        if not isinstance(header, dict) or header.get("record_type") != "header":
            raise PipelineError(
                "pre-enumerated graph manifest has no header record"
            )
        artifact = header.get("artifact_dir")
        source_record = header.get("source")
        if not isinstance(artifact, dict) or not isinstance(source_record, dict):
            raise PipelineError(
                "pre-enumerated graph manifest paths are invalid"
            )
        artifact["path"] = artifact_directory
        artifact["path_base"] = "manifest_dir"
        source_record["path"] = str(source.resolve())
        with output.open("w", encoding="utf-8") as output_stream:
            output_stream.write(
                json.dumps(
                    header, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            shutil.copyfileobj(input_stream, output_stream)


def _validated_manifest_function(
    requested: str, graph_header: Mapping[str, object]
) -> str:
    manifest_function = graph_header.get("function")
    if not isinstance(manifest_function, str) or not manifest_function:
        raise PipelineError("graph manifest function is invalid")
    if requested and requested != manifest_function:
        raise PipelineError(
            f"graph manifest function {manifest_function!r} does not match "
            f"requested function {requested!r}"
        )
    return manifest_function


def _relativize_logs(root: Path) -> None:
    """Remove the transactional directory name from published text logs."""
    prefix = str(root)
    for path in root.rglob("*.log"):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = content.replace(prefix, ".")
        if rewritten != content:
            path.write_text(rewritten, encoding="utf-8")


def _required_predictor_inputs(args: argparse.Namespace) -> Mapping[str, Any]:
    return VARIANT_DRIVER._resolve_predictor_inputs(args)


def _prepare_variant(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    graph: GraphVariant,
    root: Path,
) -> VariantRun:
    root.mkdir()
    prepared = root / "prepared.mlir"
    VARIANT_DRIVER._run(
        [
            str(optimizer),
            str(graph.ir_path),
            "--verify-each",
            f"--architecture-spec={architecture}",
            *PREPARATION_PASSES,
            "-o",
            str(prepared),
        ],
        "prepare-graph",
        root,
    )
    run = VariantRun(
        graph=graph,
        root=root,
        prepared=prepared,
        candidates=root / "candidates.jsonl",
        bound=root / "bound.mlir",
        route_expanded=root / "route-expanded.mlir",
        task_index=root / "task-dfgs.json",
        predictor_input=root / "predictor-input.json",
        costs=root / "costs.json",
        scores=root / "scores.jsonl",
        scored=root / "scored.mlir",
    )
    enumerate_option = VARIANT_DRIVER._option(
        "enumerate-analytical-task-candidates",
        (
            ("output", run.candidates),
            ("max-cgras-per-task", args.max_cgras_per_task),
            ("max-replicas-per-task", args.max_replicas_per_task),
            ("max-replica-placement-alternatives",
             args.max_replica_placement_alternatives),
            ("max-candidates", args.max_candidates_per_variant),
            ("search-policy", args.shape_search_policy),
            ("max-changed-tasks", args.max_changed_tasks),
            ("sample-candidates", args.sample_candidates),
            ("sample-seed", args.sample_seed),
            ("pipeline-structured-candidates",
             args.pipeline_structured_candidates),
            ("function", args.function_name),
            ("graph-variant-id", graph.graph_variant_id),
            ("graph-variant-sha256", graph.ir_sha256),
        ),
    )
    VARIANT_DRIVER._run(
        [
            str(optimizer),
            str(prepared),
            enumerate_option,
            f"--architecture-spec={architecture}",
            "-o",
            str(run.bound),
        ],
        "enumerate-shapes",
        root,
    )
    VARIANT_DRIVER._require_bound_trip_counts(run.candidates)
    _, run.candidate_count = VARIANT_DRIVER._candidate_header_and_count(
        run.candidates
    )
    return run


def _prepare_variant_prediction_inputs(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    predictor: Mapping[str, Path],
    run: VariantRun,
) -> None:
    tasks = VARIANT_DRIVER._candidate_task_names(run.candidates)
    bound_text = run.bound.read_text(encoding="utf-8")
    for task in tasks:
        matches = re.findall(
            rf"\btaskflow\.task\s+@{re.escape(task)}\b", bound_text
        )
        if len(matches) != 1:
            raise PipelineError(
                f"graph {run.graph.graph_variant_id} has a non-unique task {task}"
            )

    VARIANT_DRIVER._run(
        [
            str(optimizer),
            str(run.bound),
            "--insert-data-mov",
            "-o",
            str(run.route_expanded),
        ],
        "expand-task-dfg-routes",
        run.root,
    )
    task_dir = run.root / "task-dfgs"
    VARIANT_DRIVER._run(
        [
            sys.executable,
            str(predictor["extract"]),
            "--input",
            str(run.route_expanded),
            "--output-dir",
            str(task_dir),
            "--index-output",
            str(run.task_index),
            "--relative-index-paths",
        ],
        "extract-task-dfgs",
        run.root,
    )
    index = VARIANT_DRIVER._read_json_object(run.task_index, "task DFG index")
    task_arguments = VARIANT_DRIVER._task_dfg_arguments(
        index, tasks, run.task_index
    )
    analysis_timing = run.root / "analysis-timing.json"
    analysis_cache = run.root.parent.parent / "analysis-cache"
    analysis_cache.mkdir(exist_ok=True)
    VARIANT_DRIVER._run(
        [
            sys.executable,
            str(predictor["features"]),
            "--manifest",
            str(run.candidates),
            *task_arguments,
            "--neura-opt",
            str(optimizer),
            "--architecture",
            str(architecture),
            "--workload-domain",
            args.workload_domain,
            "--output",
            str(run.predictor_input),
            "--timing-output",
            str(analysis_timing),
            "--analysis-cache-dir",
            str(analysis_cache),
        ],
        "analyze-task-shapes",
        run.root,
    )


def _task_paths_for_variant(run: VariantRun) -> Mapping[str, Path]:
    tasks = VARIANT_DRIVER._candidate_task_names(run.candidates)
    index = VARIANT_DRIVER._read_json_object(run.task_index, "task DFG index")
    arguments = VARIANT_DRIVER._task_dfg_arguments(
        index, tasks, run.task_index
    )
    result = {}
    for argument in arguments:
        prefix = "--task-dfg="
        if not argument.startswith(prefix):
            raise PipelineError("task DFG argument has an invalid form")
        task, separator, raw_path = argument[len(prefix):].partition("=")
        if not separator or not task or not raw_path or task in result:
            raise PipelineError("task DFG argument has an invalid assignment")
        result[task] = Path(raw_path).resolve()
    return result


def _predict_variants_with_shared_cache(
    args: argparse.Namespace,
    predictor: Mapping[str, Any],
    runs: Sequence[VariantRun],
    stage: Path,
) -> Mapping[str, object]:
    """Generate every catalogue under the selected predictor contract."""
    if predictor["catalog_mode"] == "formal-ensemble":
        return _predict_variants_with_formal_ensemble(
            args, predictor, runs, stage
        )

    started = time.perf_counter()
    adapter = _load_catalog_driver(predictor["catalog"])
    device = adapter.torch.device(args.device)
    model, model_metadata = adapter.load_reference_checkpoint(
        predictor["checkpoint"],
        allow_test_fixture=args.allow_test_fixture,
        allow_provisional=args.allow_provisional_checkpoint,
    )
    model = model.to(device)
    model_bundle = (model, model_metadata)
    prediction_cache = {}
    dfg_cache = {}
    totals = {
        "schema": "amoeba-shared-predictor-timing-v1",
        "cache_policy": "task-body-oriented-shape-cross-graph-v1",
        "graph_catalog_count": len(runs),
        "model_load_count": 1,
        "task_dfg_parse_count": 0,
        "task_dfg_cache_hits": 0,
        "unique_cost_query_count": 0,
        "local_unique_body_shape_query_count": 0,
        "unique_body_shape_evaluation_count": 0,
        "body_shape_prediction_cache_hits": 0,
        "body_shape_prediction_cache_misses": 0,
        "steady_state_inference_ms": 0.0,
    }
    for run in runs:
        catalog, timing = adapter.generate_reference_catalog(
            run.candidates,
            run.predictor_input,
            _task_paths_for_variant(run),
            predictor["checkpoint"],
            device,
            allow_test_fixture=args.allow_test_fixture,
            allow_provisional=args.allow_provisional_checkpoint,
            model_bundle=model_bundle,
            prediction_cache=prediction_cache,
            dfg_cache=dfg_cache,
        )
        run.costs.write_text(
            json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (run.root / "inference-timing.json").write_text(
            json.dumps(timing, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "output": str(run.costs.resolve()),
            "namespace": catalog["namespace"],
            **timing,
        }
        (run.root / "predict-ii.stdout.log").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run.root / "predict-ii.stderr.log").write_text("", encoding="utf-8")
        for key in totals:
            if key in {
                "schema", "cache_policy", "graph_catalog_count",
                "model_load_count",
            }:
                continue
            totals[key] += timing[key]
    totals["body_shape_prediction_cache_entries"] = len(prediction_cache)
    totals["parsed_dfg_cache_entries"] = len(dfg_cache)
    totals["total_ms"] = (time.perf_counter() - started) * 1000.0
    (stage / "predictor-batch-timing.json").write_text(
        json.dumps(totals, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return totals


def _predict_variants_with_formal_ensemble(
    args: argparse.Namespace,
    predictor: Mapping[str, Any],
    runs: Sequence[VariantRun],
    stage: Path,
) -> Mapping[str, object]:
    """Generate formal catalogues without weakening the sealed adapter API."""
    started = time.perf_counter()
    adapter = _load_catalog_driver(predictor["catalog"])
    device = adapter.torch.device(args.device)
    totals = {
        "schema": "amoeba-formal-ensemble-batch-timing-v1",
        "catalog_mode": "formal-ensemble",
        "cache_policy": "independent-fail-closed-catalog-per-graph-v1",
        "graph_catalog_count": len(runs),
        "model_load_count": 0,
        "task_dfg_parse_count": 0,
        "task_dfg_cache_hits": 0,
        "unique_cost_query_count": 0,
        "local_unique_body_shape_query_count": 0,
        "unique_body_shape_evaluation_count": 0,
        "body_shape_prediction_cache_hits": 0,
        "body_shape_prediction_cache_misses": 0,
        "steady_state_inference_ms": 0.0,
        "catalog_total_ms": 0.0,
        "supported_query_count": 0,
        "unsupported_query_count": 0,
        "ensemble_report_sha256": sha256_file(predictor["checkpoint"]),
        "checkpoint_sha256": {
            name: sha256_file(path)
            for name, path in predictor["checkpoint_paths"].items()
        },
    }
    for run in runs:
        catalog, timing = adapter.generate_catalog(
            run.candidates,
            run.predictor_input,
            _task_paths_for_variant(run),
            predictor["checkpoint_paths"],
            predictor["checkpoint"],
            device,
            "selected",
        )
        run.costs.write_text(
            json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (run.root / "inference-timing.json").write_text(
            json.dumps(timing, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "output": str(run.costs.resolve()),
            "namespace": catalog["namespace"],
            **timing,
        }
        (run.root / "predict-ii.stdout.log").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run.root / "predict-ii.stderr.log").write_text("", encoding="utf-8")
        unique_evaluations = int(timing["unique_body_shape_evaluation_count"])
        totals["model_load_count"] += int(timing["model_load_count"])
        totals["task_dfg_parse_count"] += int(timing["task_dfg_parse_count"])
        totals["unique_cost_query_count"] += int(
            timing["unique_cost_query_count"]
        )
        totals["local_unique_body_shape_query_count"] += unique_evaluations
        totals["unique_body_shape_evaluation_count"] += unique_evaluations
        totals["body_shape_prediction_cache_misses"] += unique_evaluations
        totals["steady_state_inference_ms"] += float(
            timing["steady_state_inference_ms"]
        )
        totals["catalog_total_ms"] += float(timing["total_ms"])
        totals["supported_query_count"] += int(timing["supported_query_count"])
        totals["unsupported_query_count"] += int(
            timing["unsupported_query_count"]
        )
    totals["body_shape_prediction_cache_entries"] = totals[
        "unique_body_shape_evaluation_count"
    ]
    totals["parsed_dfg_cache_entries"] = totals["task_dfg_parse_count"]
    totals["total_ms"] = (time.perf_counter() - started) * 1000.0
    (stage / "predictor-batch-timing.json").write_text(
        json.dumps(totals, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return totals


def _recover_legacy_predictor_timing(
    runs: Sequence[VariantRun],
) -> Mapping[str, object]:
    """Describe a pre-cache prescored tree without mutating its source."""
    totals = {
        "schema": "amoeba-shared-predictor-timing-v1",
        "cache_policy": "legacy-per-graph-predictor",
        "graph_catalog_count": len(runs),
        "model_load_count": 0,
        "task_dfg_parse_count": 0,
        "unique_cost_query_count": 0,
        "local_unique_body_shape_query_count": 0,
        "unique_body_shape_evaluation_count": 0,
        "steady_state_inference_ms": 0.0,
        "total_ms": 0.0,
    }
    for run in runs:
        timing = VARIANT_DRIVER._read_json_object(
            run.root / "inference-timing.json", "legacy predictor timing"
        )
        local_unique = timing.get(
            "local_unique_body_shape_query_count",
            timing.get("unique_body_shape_evaluation_count"),
        )
        values = {
            "model_load_count": timing.get("model_load_count"),
            "task_dfg_parse_count": timing.get("task_dfg_parse_count"),
            "unique_cost_query_count": timing.get("unique_cost_query_count"),
            "local_unique_body_shape_query_count": local_unique,
            "unique_body_shape_evaluation_count": timing.get(
                "unique_body_shape_evaluation_count"
            ),
            "steady_state_inference_ms": timing.get(
                "steady_state_inference_ms"
            ),
            "total_ms": timing.get("total_ms"),
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0
            for value in values.values()
        ):
            raise PipelineError("legacy predictor timing is invalid")
        for key, value in values.items():
            totals[key] += value
    return totals


def _score_variant_candidates(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    run: VariantRun,
) -> None:
    score_option = VARIANT_DRIVER._option(
        "score-analytical-task-candidates",
        (
            ("candidates", run.candidates),
            ("cost-file", run.costs),
            ("output", run.scores),
            ("top-k", args.ranking_top_k),
            ("compact-output", "true"),
            ("communication-mode", args.communication_mode),
            ("placement-search-policy", args.placement_search_policy),
            ("activity-scheduling-policy", args.activity_scheduling_policy),
            ("activity-beam-width", args.activity_beam_width),
            ("graph-variant-id", run.graph.graph_variant_id),
            ("graph-variant-sha256", run.graph.ir_sha256),
            ("function", args.function_name),
        ),
    )
    VARIANT_DRIVER._run(
        [
            str(optimizer),
            str(run.bound),
            score_option,
            f"--architecture-spec={architecture}",
            "-o",
            str(run.scored),
        ],
        "score-all",
        run.root,
    )


def _validate_resume_contract(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    predictor: Mapping[str, Path],
    graph_header: Mapping[str, Any],
    runs: Sequence[VariantRun],
) -> Mapping[str, Any]:
    """Fail closed before reusing any prescored artifact or writing output."""
    if predictor.get("catalog_mode", "reference") != "reference":
        raise PipelineError(
            "formal-ensemble prescored resume is not supported; run a fresh "
            "transaction so the sealed ensemble validates every catalogue"
        )
    tile_factors = list(joint_graph_variant_space._selected_tile_factors(
        args.tile_factors, args.min_tile_factor, args.max_tile_factor
    ))
    expected_graph_bounds = {
        "max_depth": args.max_graph_depth,
        "max_variants": args.max_graph_variants,
        "min_tile_factor": args.min_tile_factor,
        "max_tile_factor": args.max_tile_factor,
        "tile_factors": tile_factors,
        "tile_factor_one_is_identity": 1 in tile_factors,
        "forbid_nested_tiling": args.forbid_nested_tiling,
        "max_fission_actions_per_task": args.max_fission_actions_per_task,
        "fission_fact_generation_enabled": args.max_fission_steps > 0,
        "max_duplicate_samples": args.max_duplicate_samples,
        "max_fusion_steps": args.max_fusion_steps,
        "max_fission_steps": args.max_fission_steps,
        "max_tiling_steps": args.max_tiling_steps,
    }
    if graph_header.get("bounds") != expected_graph_bounds:
        raise PipelineError("resume graph-search bounds do not match artifacts")

    adapter = _load_catalog_driver(predictor["catalog"])
    _, current_model = adapter.load_reference_checkpoint(
        predictor["checkpoint"],
        allow_test_fixture=args.allow_test_fixture,
        allow_provisional=args.allow_provisional_checkpoint,
    )
    checkpoint_sha = sha256_file(predictor["checkpoint"])
    optimizer_sha = sha256_file(optimizer)
    architecture_sha = sha256_file(architecture)
    expected_model = {
        "sha256": checkpoint_sha,
        "config_sha256": hashlib.sha256(json.dumps({
            "feature_names_sha256": current_model["feature_names_sha256"],
            "feature_contract_id": current_model["feature_contract_id"],
            "feature_extractor_source_sha256": current_model[
                "feature_extractor_source_sha256"
            ],
            "fabric": current_model["fabric"],
            "applicability": current_model["applicability"],
            "parameter_count": current_model["parameter_count"],
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )).hexdigest(),
        "quality_status": current_model["quality_status"],
        "parameter_count": current_model["parameter_count"],
        "feature_contract_id": current_model["feature_contract_id"],
        "feature_extractor_source_sha256": current_model[
            "feature_extractor_source_sha256"
        ],
        "training_evidence": current_model["training_evidence"],
    }
    expected_architecture_contract = {
        "contract_id": current_model["fabric"]["contract_id"],
        "fabric": current_model["fabric"],
        "shape_protocol": adapter.SHAPE_PROTOCOL_ID,
        "supported_architecture_sha256": [architecture_sha],
        "architecture_sha256_role":
            "catalog_provenance_for_validated_fabric",
    }
    scope_by_policy = {
        "complete-cartesian": "static-shape-cartesian-product",
        "one-change-from-unit": "static-shape-one-change-from-unit",
        "bounded-changes-from-unit":
            "static-shape-bounded-changes-from-unit",
        "projected-sample": "static-shape-projected-sample",
        "projected-sample-with-pipeline":
            "static-shape-projected-sample-with-pipeline",
    }
    normalized_contract = None
    for run in runs:
        candidate_header, _ = VARIANT_DRIVER._candidate_header_and_count(
            run.candidates
        )
        machine = candidate_header.get("architecture")
        if not isinstance(machine, Mapping):
            raise PipelineError("resume candidate architecture is invalid")
        grid_rows = machine.get("grid_rows")
        grid_cols = machine.get("grid_cols")
        if (isinstance(grid_rows, bool) or not isinstance(grid_rows, int) or
                isinstance(grid_cols, bool) or not isinstance(grid_cols, int)):
            raise PipelineError("resume candidate grid dimensions are invalid")
        effective_max_cgras = (
            grid_rows * grid_cols if args.max_cgras_per_task == 0
            else min(args.max_cgras_per_task, grid_rows * grid_cols)
        )
        contract = {
            "resource_axis_version": 3,
            "search_scope": scope_by_policy[args.shape_search_policy],
            "max_cgras_per_task": effective_max_cgras,
            "max_replicas_per_task": args.max_replicas_per_task,
            "max_replica_placement_alternatives":
                args.max_replica_placement_alternatives,
            "max_candidates": args.max_candidates_per_variant,
            "architecture_sha256": architecture_sha,
        }
        actual = {
            "resource_axis_version": candidate_header.get(
                "resource_axis_version"
            ),
            "search_scope": candidate_header.get("search_scope"),
            "max_cgras_per_task": candidate_header.get("max_cgras_per_task"),
            "max_replicas_per_task": candidate_header.get(
                "max_replicas_per_task", 1
            ),
            "max_replica_placement_alternatives": candidate_header.get(
                "max_replica_placement_alternatives", 0
            ),
            "max_candidates": (
                candidate_header.get("candidate_budget", {})
                .get("max_candidates")
                if isinstance(candidate_header.get("candidate_budget"), Mapping)
                else None
            ),
            "architecture_sha256": machine.get("spec_sha256"),
        }
        if args.shape_search_policy == "bounded-changes-from-unit":
            contract["max_changed_tasks"] = args.max_changed_tasks
            actual["max_changed_tasks"] = candidate_header.get(
                "max_changed_tasks"
            )
        if args.shape_search_policy in (
            "projected-sample", "projected-sample-with-pipeline"
        ):
            contract.update({
                "sample_candidates": args.sample_candidates,
                "sample_seed": args.sample_seed,
            })
            actual.update({
                "sample_candidates": candidate_header.get("sample_candidates"),
                "sample_seed": candidate_header.get("sample_seed"),
            })
        if args.shape_search_policy == "projected-sample-with-pipeline":
            contract["pipeline_structured_candidates"] = (
                args.pipeline_structured_candidates
            )
            actual["pipeline_structured_candidates"] = candidate_header.get(
                "pipeline_structured_candidates"
            )
        if actual != contract:
            raise PipelineError(
                f"resume candidate search contract mismatch for "
                f"{run.graph.graph_variant_id}"
            )
        if normalized_contract is None:
            normalized_contract = actual
        elif actual != normalized_contract:
            raise PipelineError("resume variants use different search contracts")

        costs = VARIANT_DRIVER._read_json_object(run.costs, "cost catalogue")
        metadata = costs.get("predictor_metadata")
        if not isinstance(metadata, Mapping):
            raise PipelineError("resume cost catalogue lacks predictor metadata")
        model = metadata.get("model")
        provenance = metadata.get("analytical_provenance")
        analytical_input = VARIANT_DRIVER._read_json_object(
            run.predictor_input, "analytical predictor input"
        )
        input_provenance = analytical_input.get("provenance")
        if not isinstance(input_provenance, Mapping):
            raise PipelineError(
                f"resume analytical input lacks provenance for "
                f"{run.graph.graph_variant_id}"
            )
        task_records = candidate_header.get("tasks")
        if not isinstance(task_records, list) or not task_records:
            raise PipelineError("resume candidate task metadata is invalid")
        task_body_sha256 = {}
        for task_record in task_records:
            if not isinstance(task_record, Mapping):
                raise PipelineError("resume candidate task metadata is invalid")
            task = task_record.get("task")
            body_sha = task_record.get("body_sha256")
            if (
                not isinstance(task, str) or not task
                or not isinstance(body_sha, str) or len(body_sha) != 64
                or task in task_body_sha256
            ):
                raise PipelineError("resume candidate task metadata is invalid")
            task_body_sha256[task] = body_sha
        task_paths = _task_paths_for_variant(run)
        task_dfg_sha256 = {
            task: sha256_file(path) for task, path in task_paths.items()
        }
        if set(task_paths) != set(task_body_sha256):
            raise PipelineError("resume task DFG mapping is incomplete")
        for task, path in task_paths.items():
            if adapter.source_task_body_sha256(
                path.read_text(encoding="utf-8"), task
            ) != task_body_sha256[task]:
                raise PipelineError(
                    f"resume task body provenance mismatch for {task}"
                )
        provenance_keys = (
            "architecture_sha256", "neura_opt_sha256", "task_body_sha256",
            "task_dfg_sha256", "rec_res_source", "startup_cycles_source",
            "applicability",
        )
        expected_provenance = {
            key: input_provenance.get(key) for key in provenance_keys
        }
        applicability = (
            provenance.get("applicability")
            if isinstance(provenance, Mapping) else None
        )
        if (
            metadata.get("candidate_manifest_sha256")
            != sha256_file(run.candidates)
            or metadata.get("analytical_input_sha256")
            != sha256_file(run.predictor_input)
            or not isinstance(model, Mapping)
            or dict(model) != expected_model
            or metadata.get("architecture_contract")
            != expected_architecture_contract
            or not isinstance(provenance, Mapping)
            or dict(provenance) != expected_provenance
            or input_provenance.get("neura_opt_sha256") != optimizer_sha
            or input_provenance.get("architecture_sha256") != architecture_sha
            or input_provenance.get("task_body_sha256") != task_body_sha256
            or input_provenance.get("task_dfg_sha256") != task_dfg_sha256
            or not isinstance(applicability, Mapping)
            or applicability.get("workload_domain") != args.workload_domain
        ):
            raise PipelineError(
                f"resume predictor provenance mismatch for "
                f"{run.graph.graph_variant_id}"
            )
    return dict(normalized_contract or {})


def _selected_score(
    args: argparse.Namespace,
    architecture: Path,
    run: VariantRun,
    candidate_id: str,
) -> Mapping[str, object]:
    shortlist = VARIANT_DRIVER._read_shortlist(
        run.scores,
        run.candidates,
        run.costs,
        architecture,
        args.communication_mode,
        args.placement_search_policy,
        args.activity_scheduling_policy,
        args.activity_beam_width,
        run.graph.graph_variant_id,
        run.graph.ir_sha256,
    )
    matches = [row for row in shortlist if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise PipelineError("selected candidate is absent from its bound score manifest")
    return matches[0]


def _replay_selected(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    run: VariantRun,
    selected: Mapping[str, object],
    stage: Path,
    rank: int,
) -> Mapping[str, object]:
    candidate_id = str(selected["candidate_id"])
    output = stage / f"selected-top-{rank}.mlir"
    mapper_cache = stage / "mapper-cache"
    mapper_cache.mkdir(exist_ok=True)
    materialize = VARIANT_DRIVER._option(
        "materialize-analytical-task-candidate",
        (
            ("candidates", run.candidates),
            ("candidate-id", candidate_id),
            ("function", args.function_name),
        ),
    )
    mapper = VARIANT_DRIVER._option(
        "map-joint-scheduling-tasks",
        (
            ("scores", run.scores),
            ("candidate-id", candidate_id),
            ("function", args.function_name),
            ("mapper-cache-dir", mapper_cache),
            ("expected-trace-sha256", selected["schedule_trace_sha256"]),
            ("expected-score-sha256", sha256_file(run.scores)),
        ),
    )
    orchestration = VARIANT_DRIVER._option(
        "orchestrate-tasks-on-accelerators",
        (
            ("orchestration-strategy", "analytical-based-task-orchestration"),
            ("scheduling-mode", "spatial-temporal"),
            ("communication-mode", args.communication_mode),
            ("placement-search-policy", args.placement_search_policy),
            ("activity-scheduling-policy", args.activity_scheduling_policy),
            ("activity-beam-width", args.activity_beam_width),
        ),
    )
    command = [
        str(optimizer),
        str(run.bound),
        materialize,
        mapper,
        orchestration,
        f"--architecture-spec={architecture}",
        "-o",
        str(output),
    ]
    label = f"replay-global-top-{rank}"
    base = {
        "graph_variant_id": run.graph.graph_variant_id,
        "graph_variant_sha256": run.graph.ir_sha256,
        "graph_actions": list(run.graph.record.get("actions", [])),
        "candidate_id": candidate_id,
        "predicted_scheduler_makespan": selected[
            "predicted_scheduler_makespan"
        ],
        "predicted_schedule_trace_sha256": selected["schedule_trace_sha256"],
    }
    if args.replay_timeout_seconds > 0:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.replay_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            (stage / f"{label}.stdout.log").write_text(stdout)
            (stage / f"{label}.stderr.log").write_text(stderr)
            return {
                **base,
                "real_mapper_status": "timeout",
                "timeout_seconds": args.replay_timeout_seconds,
            }
        (stage / f"{label}.stdout.log").write_text(completed.stdout)
        (stage / f"{label}.stderr.log").write_text(completed.stderr)
        if completed.returncode:
            return {
                **base,
                "exit_code": completed.returncode,
                "real_mapper_status": "failed",
            }
    else:
        VARIANT_DRIVER._run(command, label, stage)
    replay = output.read_text(encoding="utf-8")
    expected_tasks = len(
        VARIANT_DRIVER._candidate_shapes(run.candidates)[candidate_id]
    )
    if (
        replay.count("amoeba.mapper_replay_verified") != expected_tasks
        or "joint_scheduling_replay_verified" not in replay
        or "joint_scheduling_actual_trace_sha256" not in replay
    ):
        raise PipelineError(
            f"global top-{rank} replay did not verify every task"
        )
    makespan = re.search(
        r"joint_scheduling_actual_makespan\s*=\s*(-?[0-9]+)", replay
    )
    trace = re.search(
        r'joint_scheduling_actual_trace_sha256\s*=\s*"([0-9a-f]{64})"',
        replay,
    )
    if makespan is None or int(makespan.group(1)) <= 0 or trace is None:
        raise PipelineError(
            f"global top-{rank} replay has no valid realized trace"
        )
    cache_hits = re.search(
        r"joint_scheduling_mapper_cache_hits\s*=\s*(-?[0-9]+)", replay
    )
    cache_misses = re.search(
        r"joint_scheduling_mapper_cache_misses\s*=\s*(-?[0-9]+)", replay
    )
    if cache_hits is None or cache_misses is None:
        raise PipelineError(
            f"global top-{rank} replay has no mapper-cache counters"
        )
    return {
        **base,
        "actual_scheduler_makespan": int(makespan.group(1)),
        "actual_schedule_trace_sha256": trace.group(1),
        "mapper_cache_hits": int(cache_hits.group(1)),
        "mapper_cache_misses": int(cache_misses.group(1)),
        "output": _relative(stage, output),
        "real_mapper_status": "verified",
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    total_started = time.perf_counter()
    source = VARIANT_DRIVER._require_file(args.input, "input MLIR")
    architecture = VARIANT_DRIVER._require_file(
        args.architecture, "architecture YAML"
    )
    optimizer = VARIANT_DRIVER._require_file(
        args.amoeba_opt, "mlir-amoeba-opt", executable=True
    )
    predictor = _required_predictor_inputs(args)
    resumed = args.resume_prescored_dir is not None
    resume_source = None
    if resumed:
        resume_source = args.resume_prescored_dir.expanduser().resolve()
        final = args.output_dir.expanduser().resolve()
        if not resume_source.is_dir() or resume_source == final:
            raise PipelineError(
                "resume directory must be an existing prescored tree"
            )
        try:
            final.relative_to(resume_source)
        except ValueError:
            pass
        else:
            raise PipelineError(
                "output directory must not be inside the immutable resume tree"
            )
        final, stage = _new_output_transaction(args.output_dir)
    else:
        final, stage = _new_output_transaction(args.output_dir)
    published = False
    try:
        graph_started = time.perf_counter()
        if resumed:
            # Resume input is immutable. Work in the same disposable sibling
            # transaction used by a fresh run, so provenance or replay failure
            # cannot add files to or consume the supplied prescored tree.
            assert resume_source is not None
            shutil.copytree(resume_source, stage, dirs_exist_ok=True)
        graph_manifest = stage / "graph-variants.jsonl"
        graph_artifacts = stage / "graph-variants"
        if resumed:
            if args.graph_manifest_input is not None:
                raise PipelineError(
                    "resume cannot also accept a graph manifest input"
                )
        elif args.graph_manifest_input is None:
            graph_args = argparse.Namespace(
                input=source,
                amoeba_opt=optimizer,
                output=graph_manifest,
                artifact_dir=graph_artifacts,
                max_depth=args.max_graph_depth,
                max_variants=args.max_graph_variants,
                min_tile_factor=args.min_tile_factor,
                max_tile_factor=args.max_tile_factor,
                tile_factors=args.tile_factors,
                forbid_nested_tiling=args.forbid_nested_tiling,
                max_fission_actions_per_task=args.max_fission_actions_per_task,
                max_duplicate_samples=args.max_duplicate_samples,
                max_fusion_steps=args.max_fusion_steps,
                max_fission_steps=args.max_fission_steps,
                max_tiling_steps=args.max_tiling_steps,
                function=args.function,
                witness_command=args.witness_command,
                timeout_seconds=args.graph_timeout_seconds,
            )
            enumerate_graph_space(graph_args)
        else:
            supplied = VARIANT_DRIVER._require_file(
                args.graph_manifest_input, "pre-enumerated graph manifest"
            )
            supplied_header, _ = load_graph_variants(supplied)
            supplied_source = supplied_header.get("source", {})
            if supplied_source.get("sha256") != sha256_file(source):
                raise PipelineError(
                    "pre-enumerated graph manifest source does not match input"
                )
            artifact = supplied_header.get("artifact_dir", {})
            if (
                artifact.get("path_base") != "manifest_dir"
                or not isinstance(artifact.get("path"), str)
            ):
                raise PipelineError(
                    "pre-enumerated graph manifest artifact directory is invalid"
                )
            supplied_artifacts = (
                supplied.parent / str(artifact["path"])
            ).resolve()
            shutil.copytree(supplied_artifacts, graph_artifacts)
            _copy_graph_manifest_with_rebased_artifacts(
                supplied, graph_manifest, graph_artifacts.name, source
            )
        graph_header, graphs = load_graph_variants(graph_manifest)
        if graph_header.get("source", {}).get("sha256") != sha256_file(source):
            raise PipelineError("graph manifest source does not match input")
        if graph_header.get("optimizer", {}).get("sha256") != sha256_file(
            optimizer
        ):
            raise PipelineError(
                "graph manifest optimizer does not match this executable"
            )
        selected_graph_ids = list(getattr(args, "graph_variant_id", []))
        if selected_graph_ids:
            requested_graphs = set(selected_graph_ids)
            available_graphs = {
                graph.graph_variant_id for graph in graphs
            }
            missing_graphs = sorted(requested_graphs - available_graphs)
            if missing_graphs:
                raise PipelineError(
                    "requested graph variants are absent: "
                    + ", ".join(missing_graphs)
                )
            graphs = [
                graph for graph in graphs
                if graph.graph_variant_id in requested_graphs
            ]
        args.function_name = _validated_manifest_function(
            args.function, graph_header
        )
        graph_seconds = time.perf_counter() - graph_started

        analytical_started = time.perf_counter()
        variant_root = stage / "variants"
        if resumed:
            runs = []
            for graph in graphs:
                run_root = variant_root / (
                    f"{graph.index:06d}-{graph.graph_variant_id}"
                )
                run = VariantRun(
                    graph=graph,
                    root=run_root,
                    prepared=run_root / "prepared.mlir",
                    candidates=run_root / "candidates.jsonl",
                    bound=run_root / "bound.mlir",
                    route_expanded=run_root / "route-expanded.mlir",
                    task_index=run_root / "task-dfgs.json",
                    predictor_input=run_root / "predictor-input.json",
                    costs=run_root / "costs.json",
                    scores=run_root / "scores.jsonl",
                    scored=run_root / "scored.mlir",
                )
                required = (
                    run.prepared, run.candidates, run.bound,
                    run.route_expanded, run.task_index,
                    run.predictor_input, run.costs, run.scores, run.scored,
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise PipelineError(
                        "resumed variant is incomplete: " + ", ".join(missing)
                    )
                _, run.candidate_count = (
                    VARIANT_DRIVER._candidate_header_and_count(run.candidates)
                )
                runs.append(run)
        else:
            variant_root.mkdir()

            def prepare(graph: GraphVariant) -> VariantRun:
                run_root = variant_root / (
                    f"{graph.index:06d}-{graph.graph_variant_id}"
                )
                return _prepare_variant(
                    args, architecture, optimizer, graph, run_root
                )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                runs = list(executor.map(prepare, graphs))
        resume_contract = {}
        if resumed:
            resume_contract = _validate_resume_contract(
                args, architecture, optimizer, predictor, graph_header, runs
            )
        total_candidates = sum(run.candidate_count for run in runs)
        if total_candidates > args.max_total_candidates:
            raise PipelineError(
                "global candidate budget exceeded before predictor scoring"
            )

        if not resumed:
            def prepare_prediction_inputs(variant: VariantRun) -> None:
                _prepare_variant_prediction_inputs(
                    args, architecture, optimizer, predictor, variant
                )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(prepare_prediction_inputs, runs))

            predictor_batch = _predict_variants_with_shared_cache(
                args, predictor, runs, stage
            )

            def score_candidates(variant: VariantRun) -> None:
                _score_variant_candidates(
                    args, architecture, optimizer, variant
                )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(score_candidates, runs))
        else:
            predictor_batch_path = stage / "predictor-batch-timing.json"
            recovered_predictor_batch = False
            if predictor_batch_path.is_file():
                predictor_batch = VARIANT_DRIVER._read_json_object(
                    predictor_batch_path, "shared predictor timing"
                )
            else:
                predictor_batch = _recover_legacy_predictor_timing(runs)
                recovered_predictor_batch = True

        artifacts = [
            VariantArtifacts(
                run.graph, run.candidates, run.costs, run.scores
            )
            for run in runs
        ]
        ranking = build_global_ranking(
            graph_manifest,
            architecture,
            artifacts,
            args.communication_mode,
            args.ranking_top_k,
            args.max_total_candidates,
            args.placement_search_policy,
            args.activity_scheduling_policy,
            args.activity_beam_width,
            [graph.graph_variant_id for graph in graphs],
        )
        analytical_seconds = time.perf_counter() - analytical_started
        replay_requests = []
        if args.replay_selection:
            run_by_graph_id = {
                run.graph.graph_variant_id: run for run in runs
            }
            for rank, (graph_id, candidate_id) in enumerate(
                args.replay_selection, start=1
            ):
                selected_run = run_by_graph_id.get(graph_id)
                if selected_run is None:
                    raise PipelineError(
                        f"explicit replay graph is absent: {graph_id}"
                    )
                score = _selected_score(
                    args, architecture, selected_run, candidate_id
                )
                replay_requests.append((selected_run, score, rank))
        else:
            for rank, selected in enumerate(
                ranking["shortlist"][: args.replay_top_k], start=1
            ):
                selected_run = next(
                    run
                    for run in runs
                    if run.graph.graph_variant_id
                    == selected["graph_variant_id"]
                )
                score = _selected_score(
                    args,
                    architecture,
                    selected_run,
                    str(selected["candidate_id"]),
                )
                replay_requests.append(
                    (selected_run, score, rank)
                )

        # A resume tree is immutable input until all reused score manifests
        # have passed their full binding checks. In particular, a malformed
        # score must not leave a recovered timing file or ranking behind.
        if resumed and recovered_predictor_batch:
            predictor_batch_path.write_text(
                json.dumps(
                    predictor_batch, indent=2, sort_keys=True, allow_nan=False
                ) + "\n",
                encoding="utf-8",
            )
        ranking_path = stage / "global-ranking.json"
        write_global_ranking(ranking_path, ranking)

        def replay_selected(request: tuple[VariantRun, Mapping[str, object], int]):
            selected_run, score, rank = request
            return _replay_selected(
                args,
                architecture,
                optimizer,
                selected_run,
                score,
                stage,
                rank,
            )

        # Native mapper replay is task-local but a difficult kernel can take
        # much longer than its peers.  Run independent global top-k entries in
        # parallel so one hard candidate does not head-of-line block every
        # later rank.  executor.map preserves rank order in the final report;
        # the persistent mapper cache uses atomic publication, so concurrent
        # equivalent misses can only duplicate work, never publish partial IR.
        replay_workers = min(args.workers, len(replay_requests))
        replay_started = time.perf_counter()
        if replay_workers == 1:
            # A single worker means sequential replay, not top-1-only replay.
            # Preserve the complete requested shortlist so reports remain
            # comparable with multi-worker executions.
            replays = [replay_selected(request) for request in replay_requests]
        else:
            with ThreadPoolExecutor(max_workers=replay_workers) as executor:
                replays = list(executor.map(replay_selected, replay_requests))
        replay_seconds = time.perf_counter() - replay_started

        replay_complete = all(
            replay.get("real_mapper_status") == "verified"
            for replay in replays
        )
        report = {
            "schema": "amoeba-joint-program-report-v1",
            "complete": replay_complete,
            "analytical_scoring_complete": True,
            "input": {"path": str(source), "sha256": sha256_file(source)},
            "architecture": {
                "path": str(architecture),
                "sha256": sha256_file(architecture),
            },
            "predictor_checkpoint": {
                "path": str(predictor["checkpoint"]),
                "sha256": sha256_file(predictor["checkpoint"]),
                "provisional_allowed": args.allow_provisional_checkpoint,
                "fixture_allowed": args.allow_test_fixture,
            },
            "predictor_root": str(predictor["root"]),
            "predictor_catalog_mode": predictor["catalog_mode"],
            "predictor_component_sha256": {
                name: sha256_file(path)
                for name, path in predictor["checkpoint_paths"].items()
            },
            "communication_mode": args.communication_mode,
            "placement_search_policy": args.placement_search_policy,
            "activity_scheduling_policy": args.activity_scheduling_policy,
            "activity_beam_width": args.activity_beam_width,
            "max_cgras_per_task": args.max_cgras_per_task,
            "max_replicas_per_task": args.max_replicas_per_task,
            "max_replica_placement_alternatives":
                args.max_replica_placement_alternatives,
            "timeout_policy": {
                "graph_action_seconds": args.graph_timeout_seconds,
                "native_replay_seconds": args.replay_timeout_seconds,
                "zero_means_unbounded": True,
            },
            "native_replay_selection": {
                "mode": (
                    "explicit_graph_candidate"
                    if args.replay_selection else "global_predicted_top_k"
                ),
                "requested": [
                    {"graph_variant_id": graph_id,
                     "candidate_id": candidate_id}
                    for graph_id, candidate_id in args.replay_selection
                ],
            },
            "graph_search_bounds": graph_header.get("bounds", {}),
            "selected_graph_variant_ids": [
                graph.graph_variant_id for graph in graphs
            ],
            "resumed_from_complete_prescored_variants": resumed,
            "resume_validation": "verified" if resumed else "not-applicable",
            "prescored_contract": resume_contract if resumed else None,
            "counts": {
                "graph_variants": len(graphs),
                "total_candidates": total_candidates,
                "mapper_invocations": len(replays),
                "mapper_replay_workers": replay_workers,
                "shared_predictor": predictor_batch,
                "graph_enumeration": graph_header.get(
                    "enumeration_counts", {}
                ),
            },
            "timings_seconds": {
                "graph_enumeration_and_validation": graph_seconds,
                "analytical_preparation_scoring_and_ranking":
                    analytical_seconds,
                "native_replay": replay_seconds,
                "total_before_publication":
                    time.perf_counter() - total_started,
            },
            "artifacts": {
                "graph_manifest": _relative(stage, graph_manifest),
                "global_ranking": _relative(stage, ranking_path),
                "predictor_batch_timing": _relative(
                    stage, stage / "predictor-batch-timing.json"
                ),
            },
            "selected_top_1": replays[0],
            "selected_top_k": replays,
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _relativize_logs(stage)
        os.replace(stage, final)
        published = True
        return report
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument(
        "--amoeba-opt",
        type=Path,
        default=AMOEBA_ROOT / "build" / "tools" / "mlir-amoeba-opt" / "mlir-amoeba-opt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-manifest-input", type=Path,
        help="Reuse a complete, witnessed graph manifest after validation.",
    )
    parser.add_argument(
        "--graph-variant-id", action="append", default=[],
        help=("prepare and score only this graph ID from the validated graph "
              "manifest; repeatable"),
    )
    parser.add_argument(
        "--resume-prescored-dir", type=Path,
        help=("Validate and finalize an immutable prescored tree with complete "
              "variant scores."),
    )
    parser.add_argument("--function", default="")
    parser.add_argument("--witness-command", required=True)
    parser.add_argument("--max-graph-depth", type=int, default=3)
    parser.add_argument("--max-graph-variants", type=int, default=1000)
    parser.add_argument("--min-tile-factor", type=int, default=2)
    parser.add_argument("--max-tile-factor", type=int, default=8)
    parser.add_argument(
        "--tile-factors", default=None,
        help=("Comma-separated selected graph-tiling factors; factor 1 is "
              "the identity graph; defaults to 1,2,4,8 within the max bound."),
    )
    parser.add_argument(
        "--forbid-nested-tiling", action="store_true",
        help="Exclude tiling actions whose target is already a derived tile",
    )
    parser.add_argument("--max-fission-actions-per-task", type=int, default=256)
    parser.add_argument("--max-duplicate-samples", type=int, default=100)
    parser.add_argument("--max-fusion-steps", type=int, default=2)
    parser.add_argument(
        "--max-fission-steps", type=int, default=0,
        help=("Maximum fission rewrites in one graph variant. Fission is "
              "temporarily disabled by default; pass a positive value to "
              "opt in."),
    )
    parser.add_argument("--max-tiling-steps", type=int, default=2)
    parser.add_argument(
        "--graph-timeout-seconds", type=float, default=0.0,
        help="Bound each graph optimizer/witness call; zero means unbounded",
    )
    parser.add_argument(
        "--max-cgras-per-task", type=int, default=0,
        help=("maximum physical CGRAs per replica; the aggregate replica "
              "footprint must still fit the grid; zero uses the full grid"),
    )
    parser.add_argument(
        "--max-replicas-per-task", type=int, choices=(1, 2, 4, 8), default=1,
        help="maximum replicas for statically eligible data-parallel tasks",
    )
    parser.add_argument(
        "--max-replica-placement-alternatives", type=int, default=0,
        help=("prune a multi-replica shape choice when its canonical legal "
              "replica-set placement count exceeds this value; zero is "
              "unbounded"),
    )
    parser.add_argument(
        "--shape-search-policy",
        choices=("complete-cartesian", "one-change-from-unit",
                 "bounded-changes-from-unit", "projected-sample",
                 "projected-sample-with-pipeline"),
        default="projected-sample",
    )
    parser.add_argument("--max-changed-tasks", type=int, default=2)
    parser.add_argument("--sample-candidates", type=int, default=20_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--pipeline-structured-candidates", type=int, default=4,
        help=("number of pipeline-structured resource points reserved inside "
              "the fixed projected sample budget"),
    )
    parser.add_argument("--max-candidates-per-variant", type=int, default=1_000_000)
    parser.add_argument("--max-total-candidates", type=int, default=1_000_000)
    parser.add_argument("--ranking-top-k", type=int, default=1)
    parser.add_argument("--replay-top-k", type=int, default=1)
    parser.add_argument(
        "--replay-selection", action="append", default=[],
        type=_parse_replay_selection,
        metavar="GRAPH_VARIANT_ID:CANDIDATE_ID",
        help=("Replay an explicit scored graph/candidate for a controlled "
              "ablation instead of the global predicted top-k; repeatable"),
    )
    parser.add_argument(
        "--replay-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Bound each real-mapper replay; zero keeps the fail-closed "
            "unbounded behavior. Timeouts are recorded in the final report."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--workload-domain", choices=("static_tensor",), required=True
    )
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--predictor-root", type=Path, default=PREDICTOR_ROOT,
        help=("Predictor source/adapter root. The default is the pinned "
              "submodule; an explicit root permits evaluation of a sealed "
              "candidate package without replacing models/final."),
    )
    parser.add_argument(
        "--predictor-catalog-mode",
        choices=("auto", "reference", "formal-ensemble"), default="auto",
        help=("Select the adapter contract; auto recognizes the formal "
              "ensemble schema and otherwise requires a reference checkpoint."),
    )
    parser.add_argument("--allow-provisional-checkpoint", action="store_true")
    parser.add_argument("--allow-test-fixture", action="store_true")
    parser.add_argument(
        "--communication-mode", choices=("none", "explicit"), default="none"
    )
    parser.add_argument(
        "--placement-search-policy", choices=("exact", "heuristic"),
        default="exact",
        help=("use exact fixed-order placement or the dependency-ready "
              "communication-aware greedy placer"),
    )
    parser.add_argument(
        "--activity-scheduling-policy",
        choices=("fixed", "critical-path", "pipeline-aware", "beam", "exact"),
        default="fixed",
        help="select dependency-ready activities jointly with their bindings",
    )
    parser.add_argument("--activity-beam-width", type=int, default=1)
    args = parser.parse_args(argv)
    integer_bounds = (
        args.max_graph_variants,
        args.min_tile_factor,
        args.max_tile_factor,
        args.max_fission_actions_per_task,
        args.max_candidates_per_variant,
        args.max_total_candidates,
        args.ranking_top_k,
        args.replay_top_k,
        args.workers,
    )
    if args.max_graph_depth < 0 or any(value <= 0 for value in integer_bounds):
        parser.error("all search bounds must be positive except graph depth")
    if args.activity_beam_width <= 0:
        parser.error("--activity-beam-width must be positive")
    if args.max_duplicate_samples < 0:
        parser.error("--max-duplicate-samples must be nonnegative")
    if args.min_tile_factor < 2 or args.min_tile_factor > args.max_tile_factor:
        parser.error("tile factor bounds must satisfy 2 <= min <= max")
    if args.replay_timeout_seconds < 0:
        parser.error("--replay-timeout-seconds must be nonnegative")
    if args.replay_top_k > args.ranking_top_k:
        parser.error("--replay-top-k cannot exceed --ranking-top-k")
    if len(set(args.replay_selection)) != len(args.replay_selection):
        parser.error("--replay-selection entries must be unique")
    if len(set(args.graph_variant_id)) != len(args.graph_variant_id):
        parser.error("--graph-variant-id entries must be unique")
    if args.replay_selection and args.graph_variant_id and any(
            graph_id not in set(args.graph_variant_id)
            for graph_id, _ in args.replay_selection):
        parser.error("replay selections must be inside --graph-variant-id")
    if any(value < 0 for value in (
        args.max_fusion_steps,
        args.max_fission_steps,
        args.max_tiling_steps,
    )):
        parser.error("per-transform step limits must be nonnegative")
    if (args.max_cgras_per_task < 0 or
            args.max_replica_placement_alternatives < 0 or
            args.graph_timeout_seconds < 0):
        parser.error("invalid task-shape or timeout bound")
    if (args.shape_search_policy == "bounded-changes-from-unit" and
            args.max_changed_tasks <= 0):
        parser.error("--max-changed-tasks must be positive for bounded search")
    if (args.shape_search_policy in (
            "projected-sample", "projected-sample-with-pipeline") and
            (args.sample_candidates <= 0 or args.sample_seed < 0 or
             args.sample_candidates > args.max_candidates_per_variant)):
        parser.error("projected sample bounds are invalid")
    if (args.pipeline_structured_candidates < 0 or
            (args.shape_search_policy == "projected-sample-with-pipeline" and
             (args.pipeline_structured_candidates <= 0 or
              args.pipeline_structured_candidates >= args.sample_candidates))):
        parser.error("pipeline structured candidate bounds are invalid")
    return args


def main() -> int:
    if "--semantic-joint-tiling-fusion" in sys.argv[1:]:
        from orbit_joint_mlir_pass_entry import main as joint_main
        arguments = [arg for arg in sys.argv[1:]
                     if arg != "--semantic-joint-tiling-fusion"]
        return joint_main(arguments)
    # Source-bound semantic task tiling is an opt-in graph-candidate mode.
    # It selects and emits Taskflow IR without invoking the analytical
    # shape/replica, predictor, mapper, or hardware replay pipeline.
    if "--semantic-gemm-task-variants" in sys.argv[1:]:
        from orbit_semantic_graph_candidates import main as semantic_main
        arguments = [arg for arg in sys.argv[1:]
                     if arg != "--semantic-gemm-task-variants"]
        return semantic_main(arguments)
    try:
        report = run(parse_args())
    except (GraphSpaceError, RankingError, PipelineError, OSError, ValueError) as error:
        print(f"joint program scheduling failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
