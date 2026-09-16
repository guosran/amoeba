#!/usr/bin/env python3
"""Rank a frozen spatial-binding candidate set, then replay the top entries.

The predictor is loaded exclusively from Amoeba's pinned
``thirdparty/cgra-ii-predictor`` submodule. The input must already contain
Taskflow tasks with one pre-mapper Neura kernel per task; frontend lowering is
program-specific and remains outside this shape-only driver.

Arguments after ``--`` are forwarded to ``mlir-amoeba-opt`` after candidate
materialization. They therefore execute once per validated shortlist entry,
never once per enumerated candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional, Sequence


AMOEBA_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_ROOT = AMOEBA_ROOT / "thirdparty" / "cgra-ii-predictor"
ADAPTER_ROOT = PREDICTOR_ROOT / "adapters"
CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates"
COST_SCHEMA = "amoeba-task-shape-cost"
SCORE_SCHEMA = "amoeba-analytical-task-scores"
SCORE_MODEL = "spatial-temporal-scheduler-makespan"


class PipelineError(RuntimeError):
    """A checked orchestration step failed."""


def _sha256_file(path: Path) -> str:
    """Fingerprint the exact bytes consumed by an artifact stage.

    The hashes in the candidate, cost, and score headers are content
    identities, so this intentionally reads raw bytes rather than parsing and
    re-serializing JSON.  That preserves record order and serialization
    choices as part of the provenance chain.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"{label} does not exist: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PipelineError(f"{label} is not executable: {resolved}")
    return resolved


def _prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise PipelineError(f"output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run(command: Sequence[str], label: str, output: Path,
         cwd: Optional[Path] = None) -> None:
    """Run one checked command and preserve both output streams."""
    completed = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False,
    )
    (output / f"{label}.stdout.log").write_text(completed.stdout)
    (output / f"{label}.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise PipelineError(
            f"{label} failed with exit code {completed.returncode}: {detail}"
        )


def _task_dfg_arguments(index: Mapping[str, object],
                        expected_tasks: Sequence[str]) -> List[str]:
    result: List[str] = []
    if len(set(expected_tasks)) != len(expected_tasks):
        raise PipelineError("selected function has duplicate task names")
    for task in sorted(expected_tasks):
        raw_path = index.get(task)
        if not isinstance(task, str) or not task or not isinstance(raw_path, str):
            raise PipelineError(
                f"DFG extraction did not produce selected task {task}"
            )
        path = _require_file(Path(raw_path), f"DFG for task {task}")
        result.append(f"--task-dfg={task}={path}")
    return result


def _candidate_task_names(path: Path) -> List[str]:
    header, _ = _candidate_header_and_count(path)
    raw_tasks = header.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise PipelineError("candidate manifest has no task metadata")
    tasks = []
    for record in raw_tasks:
        if not isinstance(record, dict) or not isinstance(record.get("task"), str):
            raise PipelineError("candidate manifest task metadata is invalid")
        tasks.append(record["task"])
    return tasks


def _candidate_shapes(path: Path) -> Mapping[str, Sequence[Mapping[str, object]]]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("record_type") != "candidate":
            continue
        candidate_id = record.get("candidate_id")
        task_shapes = record.get("task_shapes")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in result
            or not isinstance(task_shapes, list)
            or not task_shapes
        ):
            raise PipelineError("candidate shape records are invalid")
        shapes = []
        for task_shape in task_shapes:
            shape = task_shape.get("shape") if isinstance(task_shape, dict) else None
            if (
                not isinstance(shape, dict)
                or isinstance(shape.get("cgra_count"), bool)
                or not isinstance(shape.get("cgra_count"), int)
                or shape["cgra_count"] <= 0
            ):
                raise PipelineError("candidate shape record is invalid")
            shapes.append(shape)
        result[candidate_id] = shapes
    return result


def _requests_pass(arguments: Sequence[str], pass_name: str) -> bool:
    option = f"--{pass_name}"
    return any(argument == option or argument.startswith(option + "=")
               for argument in arguments)


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{label} is not a JSON object")
    return value


def _candidate_header_and_count(path: Path) -> tuple[Mapping[str, object], int]:
    header = None
    footer = None
    candidate_count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"invalid candidate JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict) or record.get("schema") != CANDIDATE_SCHEMA:
            raise PipelineError("candidate manifest schema mismatch")
        kind = record.get("record_type")
        if kind == "header":
            if header is not None or candidate_count or footer is not None:
                raise PipelineError("candidate manifest header is misplaced")
            header = record
        elif kind == "candidate":
            if header is None or footer is not None:
                raise PipelineError("candidate record is outside header/footer")
            candidate_count += 1
        elif kind == "footer":
            if header is None or footer is not None:
                raise PipelineError("candidate manifest footer is misplaced")
            footer = record
        else:
            raise PipelineError("candidate manifest has an unknown record type")
    if header is None or footer is None or footer.get("candidate_count") != candidate_count:
        raise PipelineError("candidate manifest is incomplete")
    return header, candidate_count


def _require_bound_trip_counts(path: Path) -> None:
    header, _ = _candidate_header_and_count(path)
    tasks = header.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PipelineError("candidate manifest has no task metadata")
    unbound = []
    for task in tasks:
        if not isinstance(task, dict):
            raise PipelineError("candidate manifest task metadata is invalid")
        if task.get("trip_count_kind") == "symbol_dynamic":
            unbound.append(str(task.get("task", "<unnamed>")))
        elif isinstance(task.get("trip_count"), bool) or not isinstance(
            task.get("trip_count"), int
        ) or int(task["trip_count"]) <= 0:
            raise PipelineError("candidate manifest trip count is invalid")
    if unbound:
        raise PipelineError(
            "program-level joint ranking requires concrete trip counts; "
            "symbol-dynamic tasks: " + ", ".join(unbound)
        )


def _read_shortlist(path: Path, candidate_manifest: Path,
                    cost_catalog: Path,
                    architecture: Path) -> List[Mapping[str, object]]:
    """Validate the complete score artifact chain before replaying any ID."""
    header = None
    footer = None
    scores = {}
    saw_footer = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"invalid score JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict) or record.get("schema") != SCORE_SCHEMA:
            raise PipelineError("score manifest schema mismatch")
        kind = record.get("record_type")
        if kind == "header":
            if header is not None or scores or footer is not None:
                raise PipelineError("score manifest header is misplaced")
            header = record
        elif kind == "score":
            if header is None or saw_footer:
                raise PipelineError("score record is outside header/footer")
            candidate_id = record.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in scores:
                raise PipelineError("score candidate IDs are invalid or duplicated")
            scores[candidate_id] = record
        elif kind == "footer":
            if header is None or saw_footer:
                raise PipelineError("score manifest footer is misplaced")
            footer = record
            saw_footer = True
        else:
            raise PipelineError("score manifest has an unknown record type")
    if header is None or footer is None:
        raise PipelineError("score manifest is incomplete")

    candidate_header, candidate_count = _candidate_header_and_count(
        candidate_manifest
    )
    catalog = _read_json_object(cost_catalog, "cost catalogue")
    if catalog.get("schema") != COST_SCHEMA:
        raise PipelineError("cost catalogue schema mismatch")
    architecture_sha = _sha256_file(architecture)
    candidate_architecture = candidate_header.get("architecture")
    if not isinstance(candidate_architecture, dict):
        raise PipelineError("candidate manifest architecture metadata is invalid")
    # The score header must point to the exact candidate JSONL, cost JSON, and
    # architecture YAML that this driver is about to trust.  C++ has already
    # compared the architecture identity while scoring; the driver repeats the
    # artifact checks at this final handoff so a stale score file cannot replay
    # a new shortlist.
    if (
        header.get("candidate_schema") != CANDIDATE_SCHEMA
        or header.get("score_model") != SCORE_MODEL
        or header.get("candidate_manifest_sha256")
        != _sha256_file(candidate_manifest)
        or header.get("cost_catalog_sha256") != _sha256_file(cost_catalog)
        or header.get("architecture_sha256") != architecture_sha
        or candidate_architecture.get("spec_sha256") != architecture_sha
        or header.get("cost_namespace") != catalog.get("namespace")
        or header.get("function") != candidate_header.get("function")
    ):
        raise PipelineError("score manifest provenance does not match its inputs")
    valid_scores = {
        candidate_id: record for candidate_id, record in scores.items()
        if record.get("valid") is True
    }
    if (
        footer.get("candidate_count") != candidate_count
        or footer.get("scored_count") != candidate_count
        or len(scores) != candidate_count
        or footer.get("valid_count") != len(valid_scores)
    ):
        raise PipelineError("score manifest counts do not cover every candidate")

    shortlist = footer.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise PipelineError("ML scoring produced an empty shortlist")
    selected = set()
    for expected_rank, item in enumerate(shortlist):
        if not isinstance(item, dict) or item.get("rank") != expected_rank:
            raise PipelineError("score shortlist ranks are not contiguous")
        candidate_id = item.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"candidate-[0-9]+", candidate_id) is None
            or candidate_id in selected
            or candidate_id not in valid_scores
            or item.get("predicted_scheduler_makespan")
            != valid_scores[candidate_id].get("predicted_scheduler_makespan")
        ):
            raise PipelineError("score shortlist is inconsistent with valid scores")
        selected.add(candidate_id)
    return shortlist


def _option(name: str, values: Iterable[tuple[str, object]]) -> str:
    fields = [
        f"{key}={value}" for key, value in values if value not in (None, "")
    ]
    return f"--{name}=" + " ".join(fields)


def run(args: argparse.Namespace) -> Mapping[str, object]:
    input_path = _require_file(args.input, "input MLIR")
    architecture = _require_file(args.architecture, "architecture YAML")
    amoeba_opt = _require_file(args.amoeba_opt, "mlir-amoeba-opt", executable=True)
    output = _prepare_output(args.output_dir)

    required_adapters = {
        "extract": ADAPTER_ROOT / "extract_amoeba_task_dfgs.py",
        "features": ADAPTER_ROOT / "generate_amoeba_query_features.py",
        "catalog": ADAPTER_ROOT / "amoeba_cost_catalog.py",
    }
    missing = [str(path) for path in required_adapters.values() if not path.is_file()]
    if missing:
        raise PipelineError(
            "predictor submodule is not initialized; run `git submodule update "
            "--init thirdparty/cgra-ii-predictor`; missing: " + ", ".join(missing)
        )

    manifest = output / "candidates.jsonl"
    bound_ir = output / "bound.mlir"
    task_dir = output / "task-dfgs"
    task_index = output / "task-dfgs.json"
    predictor_input = output / "predictor-input.json"
    analysis_timing = output / "analysis-timing.json"
    cost_catalog = output / "costs.json"
    inference_timing = output / "inference-timing.json"
    scores = output / "scores.jsonl"
    scored_ir = output / "scored.mlir"

    enumerate_option = _option(
        "enumerate-analytical-task-candidates",
        (
            ("output", manifest),
            ("max-cgras-per-task", args.max_cgras_per_task),
            ("max-candidates", args.max_candidates),
            ("function", args.function),
        ),
    )
    _run(
        [str(amoeba_opt), str(input_path), "--classify-task-and-counter",
         enumerate_option, f"--architecture-spec={architecture}",
         "-o", str(bound_ir)],
        "enumerate", output,
    )
    _require_bound_trip_counts(manifest)

    selected_tasks = _candidate_task_names(manifest)
    bound_text = bound_ir.read_text()
    for task in selected_tasks:
        occurrences = len(re.findall(
            rf"\btaskflow\.task\s+@{re.escape(task)}\b", bound_text
        ))
        if occurrences != 1:
            raise PipelineError(
                f"selected task name {task} occurs {occurrences} times in the "
                "module; function-scoped extraction requires globally unique "
                "task names"
            )

    extract_command = [
        sys.executable, str(required_adapters["extract"]),
        "--input", str(bound_ir), "--output-dir", str(task_dir),
        "--index-output", str(task_index),
    ]
    _run(extract_command, "extract-task-dfgs", output)
    index = _read_json_object(task_index, "task DFG index")
    task_arguments = _task_dfg_arguments(index, selected_tasks)

    _run(
        [sys.executable, str(required_adapters["features"]),
         "--manifest", str(manifest), *task_arguments,
         "--neura-opt", str(amoeba_opt), "--architecture", str(architecture),
         "--output", str(predictor_input),
         "--timing-output", str(analysis_timing)],
        "analyze-task-shapes", output,
    )
    _run(
        [sys.executable, str(required_adapters["catalog"]),
         "--manifest", str(manifest),
         "--analytical-input", str(predictor_input), *task_arguments,
         "--output", str(cost_catalog),
         "--timing-output", str(inference_timing),
         "--device", args.device],
        "predict-ii", output,
    )

    score_option = _option(
        "score-analytical-task-candidates",
        (("candidates", manifest), ("cost-file", cost_catalog),
         ("output", scores), ("top-k", args.top_k),
         ("function", args.function)),
    )
    _run(
        [str(amoeba_opt), str(bound_ir), score_option,
         f"--architecture-spec={architecture}", "-o", str(scored_ir)],
        "score", output,
    )

    downstream = list(args.downstream_args)
    shortlist = _read_shortlist(scores, manifest, cost_catalog, architecture)
    shapes_by_candidate = _candidate_shapes(manifest)
    mapper_requested = _requests_pass(downstream, "map-to-accelerator")
    resource_binding_requested = _requests_pass(
        downstream, "orchestrate-tasks-on-accelerators"
    )
    for item in shortlist:
        candidate_id = str(item["candidate_id"])
        shapes = shapes_by_candidate.get(candidate_id)
        if shapes is None:
            raise PipelineError("shortlisted candidate has no shape record")
        if mapper_requested and any(shape["cgra_count"] != 1 for shape in shapes):
            raise PipelineError(
                "per-task full mapping for multi-CGRA shapes is not "
                "implemented; refusing to run the module-level mapper with "
                f"candidate {candidate_id}"
            )
    shortlist_root = output / "shortlist"
    shortlist_root.mkdir()
    materialized = []
    for item in shortlist:
        rank = int(item["rank"])
        candidate_id = str(item["candidate_id"])
        candidate_root = shortlist_root / f"{rank:03d}-{candidate_id}"
        candidate_root.mkdir()
        candidate_output = candidate_root / "output.mlir"
        materialize_option = _option(
            "materialize-analytical-task-candidate",
            (("candidates", manifest), ("candidate-id", candidate_id),
             ("function", args.function)),
        )
        _run(
            [str(amoeba_opt), str(bound_ir), materialize_option,
             f"--architecture-spec={architecture}", *downstream,
             "-o", str(candidate_output)],
            f"top-{rank}-{candidate_id}", output, cwd=candidate_root,
        )
        materialized.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "predicted_scheduler_makespan": item.get(
                "predicted_scheduler_makespan"
            ),
            "output": str(candidate_output),
        })

    report = {
        "input": str(input_path),
        "architecture": str(architecture),
        "predictor_submodule": str(PREDICTOR_ROOT),
        "predictor_status": "pending_review",
        "replay_contract": {
            "full_mapper_status": (
                "completed_single_cgra" if mapper_requested else "not_requested"
            ),
            "multi_cgra_full_mapping_supported": False,
            "resource_binding_status": (
                "completed" if resource_binding_requested else "not_requested"
            ),
        },
        "top_k_requested": args.top_k,
        "shortlist_materialization_invocations": len(materialized),
        "downstream_arguments": downstream,
        "artifacts": {
            "candidate_manifest": str(manifest),
            "bound_ir": str(bound_ir),
            "task_dfg_index": str(task_index),
            "predictor_input": str(predictor_input),
            "analysis_timing": str(analysis_timing),
            "cost_catalog": str(cost_catalog),
            "inference_timing": str(inference_timing),
            "scores": str(scores),
            "scored_ir": str(scored_ir),
        },
        "shortlist": materialized,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    downstream_arguments: List[str] = []
    if "--" in raw_arguments:
        separator = raw_arguments.index("--")
        downstream_arguments = raw_arguments[separator + 1:]
        raw_arguments = raw_arguments[:separator]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument(
        "--amoeba-opt", type=Path,
        default=AMOEBA_ROOT / "build" / "tools" / "mlir-amoeba-opt" /
        "mlir-amoeba-opt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--function", default="")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--max-cgras-per-task", type=int, default=0,
        help="maximum physical CGRAs in one task shape; zero uses the full grid",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=1_000_000,
        help="fail instead of publishing a partial manifest above this count",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(raw_arguments)
    args.downstream_args = downstream_arguments
    if args.top_k <= 0:
        parser.error("--top-k must be positive for shortlist replay")
    if args.max_cgras_per_task < 0 or args.max_candidates <= 0:
        parser.error(
            "--max-cgras-per-task must be nonnegative and "
            "--max-candidates must be positive"
        )
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, PipelineError, ValueError) as error:
        print(f"joint scheduling failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
