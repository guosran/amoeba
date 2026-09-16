#!/usr/bin/env python3
"""Run transformation-aware joint scheduling as one atomic program search.

The pipeline enumerates witnessed task-graph variants first, prepares every
variant under one versioned lowering contract, freezes every task-shape
Cartesian product, and checks the global candidate budget before invoking the
predictor.  It then scores every candidate, ranks the union, and runs the real
per-task mapper plus resource binder only for the global top-1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, List, Mapping, Optional, Sequence

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


def _required_predictor_inputs(args: argparse.Namespace) -> Mapping[str, Path]:
    checkpoint = VARIANT_DRIVER._require_file(
        args.predictor_checkpoint, "predictor checkpoint"
    )
    adapters = {
        "extract": ADAPTER_ROOT / "extract_amoeba_task_dfgs.py",
        "features": ADAPTER_ROOT / "generate_amoeba_query_features.py",
        "catalog": ADAPTER_ROOT / "amoeba_reference_cost_catalog.py",
    }
    missing = [str(path) for path in adapters.values() if not path.is_file()]
    if missing:
        raise PipelineError(
            "predictor submodule is not initialized; missing: "
            + ", ".join(missing)
        )
    return {"checkpoint": checkpoint, **adapters}


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
            ("max-candidates", args.max_candidates_per_variant),
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


def _score_variant(
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
        ],
        "analyze-task-shapes",
        run.root,
    )
    inference_timing = run.root / "inference-timing.json"
    VARIANT_DRIVER._run(
        [
            sys.executable,
            str(predictor["catalog"]),
            "--manifest",
            str(run.candidates),
            "--analytical-input",
            str(run.predictor_input),
            *task_arguments,
            "--output",
            str(run.costs),
            "--timing-output",
            str(inference_timing),
            "--device",
            args.device,
            "--checkpoint",
            str(predictor["checkpoint"]),
            *(
                ["--allow-provisional-checkpoint"]
                if args.allow_provisional_checkpoint
                else []
            ),
            *(["--allow-test-fixture"] if args.allow_test_fixture else []),
        ],
        "predict-ii",
        run.root,
    )
    score_option = VARIANT_DRIVER._option(
        "score-analytical-task-candidates",
        (
            ("candidates", run.candidates),
            ("cost-file", run.costs),
            ("output", run.scores),
            ("top-k", 0),
            ("communication-mode", args.communication_mode),
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
        run.graph.graph_variant_id,
        run.graph.ir_sha256,
    )
    matches = [row for row in shortlist if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise PipelineError("global top-1 is absent from its bound score manifest")
    return matches[0]


def _replay_top_one(
    args: argparse.Namespace,
    architecture: Path,
    optimizer: Path,
    run: VariantRun,
    selected: Mapping[str, object],
    stage: Path,
) -> Mapping[str, object]:
    candidate_id = str(selected["candidate_id"])
    output = stage / "selected-top-1.mlir"
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
        ),
    )
    VARIANT_DRIVER._run(
        [
            str(optimizer),
            str(run.bound),
            materialize,
            mapper,
            orchestration,
            f"--architecture-spec={architecture}",
            "-o",
            str(output),
        ],
        "replay-global-top-1",
        stage,
    )
    replay = output.read_text(encoding="utf-8")
    expected_tasks = len(
        VARIANT_DRIVER._candidate_shapes(run.candidates)[candidate_id]
    )
    if (
        replay.count("amoeba.mapper_replay_verified") != expected_tasks
        or "joint_scheduling_replay_verified" not in replay
        or "joint_scheduling_actual_trace_sha256" not in replay
    ):
        raise PipelineError("global top-1 replay did not verify every task")
    makespan = re.search(
        r"joint_scheduling_actual_makespan\s*=\s*(-?[0-9]+)", replay
    )
    trace = re.search(
        r'joint_scheduling_actual_trace_sha256\s*=\s*"([0-9a-f]{64})"',
        replay,
    )
    if makespan is None or int(makespan.group(1)) <= 0 or trace is None:
        raise PipelineError("global top-1 replay has no valid realized trace")
    return {
        "graph_variant_id": run.graph.graph_variant_id,
        "graph_variant_sha256": run.graph.ir_sha256,
        "candidate_id": candidate_id,
        "predicted_scheduler_makespan": selected[
            "predicted_scheduler_makespan"
        ],
        "predicted_schedule_trace_sha256": selected["schedule_trace_sha256"],
        "actual_scheduler_makespan": int(makespan.group(1)),
        "actual_schedule_trace_sha256": trace.group(1),
        "output": _relative(stage, output),
        "real_mapper_status": "verified",
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    source = VARIANT_DRIVER._require_file(args.input, "input MLIR")
    architecture = VARIANT_DRIVER._require_file(
        args.architecture, "architecture YAML"
    )
    optimizer = VARIANT_DRIVER._require_file(
        args.amoeba_opt, "mlir-amoeba-opt", executable=True
    )
    predictor = _required_predictor_inputs(args)
    final, stage = _new_output_transaction(args.output_dir)
    published = False
    try:
        graph_manifest = stage / "graph-variants.jsonl"
        graph_artifacts = stage / "graph-variants"
        graph_args = argparse.Namespace(
            input=source,
            amoeba_opt=optimizer,
            output=graph_manifest,
            artifact_dir=graph_artifacts,
            max_depth=args.max_graph_depth,
            max_variants=args.max_graph_variants,
            max_tile_factor=args.max_tile_factor,
            max_fission_actions_per_task=args.max_fission_actions_per_task,
            function=args.function,
            witness_command=args.witness_command,
            timeout_seconds=args.graph_timeout_seconds,
        )
        enumerate_graph_space(graph_args)
        graph_header, graphs = load_graph_variants(graph_manifest)
        args.function_name = str(graph_header["function"])

        variant_root = stage / "variants"
        variant_root.mkdir()
        runs: List[VariantRun] = []
        total_candidates = 0
        for graph in graphs:
            run_root = variant_root / (
                f"{graph.index:06d}-{graph.graph_variant_id}"
            )
            variant = _prepare_variant(
                args, architecture, optimizer, graph, run_root
            )
            total_candidates += variant.candidate_count
            if total_candidates > args.max_total_candidates:
                raise PipelineError(
                    "global candidate budget exceeded before predictor scoring"
                )
            runs.append(variant)

        for variant in runs:
            _score_variant(args, architecture, optimizer, predictor, variant)

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
        )
        ranking_path = stage / "global-ranking.json"
        write_global_ranking(ranking_path, ranking)
        top = ranking["shortlist"][0]
        selected_run = next(
            run
            for run in runs
            if run.graph.graph_variant_id == top["graph_variant_id"]
        )
        score = _selected_score(
            args, architecture, selected_run, str(top["candidate_id"])
        )
        replay = _replay_top_one(
            args, architecture, optimizer, selected_run, score, stage
        )

        report = {
            "schema": "amoeba-joint-program-report-v1",
            "complete": True,
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
            "communication_mode": args.communication_mode,
            "counts": {
                "graph_variants": len(graphs),
                "total_candidates": total_candidates,
                "mapper_invocations": 1,
            },
            "artifacts": {
                "graph_manifest": _relative(stage, graph_manifest),
                "global_ranking": _relative(stage, ranking_path),
            },
            "selected_top_1": replay,
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
    parser.add_argument("--function", default="")
    parser.add_argument("--witness-command", required=True)
    parser.add_argument("--max-graph-depth", type=int, default=3)
    parser.add_argument("--max-graph-variants", type=int, default=1000)
    parser.add_argument("--max-tile-factor", type=int, default=8)
    parser.add_argument("--max-fission-actions-per-task", type=int, default=256)
    parser.add_argument("--graph-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-cgras-per-task", type=int, default=0)
    parser.add_argument("--max-candidates-per-variant", type=int, default=1_000_000)
    parser.add_argument("--max-total-candidates", type=int, default=1_000_000)
    parser.add_argument("--ranking-top-k", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--workload-domain", choices=("static_tensor",), required=True
    )
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--allow-provisional-checkpoint", action="store_true")
    parser.add_argument("--allow-test-fixture", action="store_true")
    parser.add_argument(
        "--communication-mode", choices=("none", "explicit"), default="none"
    )
    args = parser.parse_args(argv)
    integer_bounds = (
        args.max_graph_variants,
        args.max_tile_factor,
        args.max_fission_actions_per_task,
        args.max_candidates_per_variant,
        args.max_total_candidates,
        args.ranking_top_k,
    )
    if args.max_graph_depth < 0 or any(value <= 0 for value in integer_bounds):
        parser.error("all search bounds must be positive except graph depth")
    if args.max_cgras_per_task < 0 or args.graph_timeout_seconds <= 0:
        parser.error("invalid task-shape or timeout bound")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (GraphSpaceError, RankingError, PipelineError, OSError, ValueError) as error:
        print(f"joint program scheduling failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
