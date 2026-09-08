#!/usr/bin/env python3
"""Enumerate, ML-score, and run the real pipeline only for top-k candidates.

The predictor is loaded exclusively from Amoeba's pinned
``thirdparty/cgra-ii-predictor`` submodule. The input must already contain
Taskflow tasks with one pre-mapper Neura kernel per task; frontend lowering is
program-specific and is intentionally kept outside this shape-only driver.

Arguments after ``--`` are forwarded to ``mlir-amoeba-opt`` after candidate
materialization. They are therefore executed exactly once per shortlisted
candidate, never once per enumerated candidate.
"""

from __future__ import annotations

import argparse
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


class PipelineError(RuntimeError):
    """A checked orchestration step failed."""


def _require_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"{label} does not exist: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PipelineError(f"{label} is not executable: {resolved}")
    return resolved


def _prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
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


def _task_dfg_arguments(index: Mapping[str, object]) -> List[str]:
    result: List[str] = []
    if not index:
        raise PipelineError("task DFG extraction produced an empty index")
    for task, raw_path in sorted(index.items()):
        if not isinstance(task, str) or not task or not isinstance(raw_path, str):
            raise PipelineError("task DFG index must map task names to paths")
        path = _require_file(Path(raw_path), f"DFG for task {task}")
        result.append(f"--task-dfg={task}={path}")
    return result


def _read_shortlist(path: Path) -> List[Mapping[str, object]]:
    footer = None
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PipelineError(
                    f"invalid score JSONL at line {line_number}: {error}"
                ) from error
            if record.get("record_type") == "footer":
                if footer is not None:
                    raise PipelineError("score manifest contains two footers")
                footer = record
    if footer is None:
        raise PipelineError("score manifest has no footer")
    shortlist = footer.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise PipelineError("ML scoring produced an empty shortlist")
    for expected_rank, item in enumerate(shortlist):
        if not isinstance(item, dict) or item.get("rank") != expected_rank:
            raise PipelineError("score shortlist ranks are not contiguous")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or re.fullmatch(
            r"candidate-[0-9]+", candidate_id,
        ) is None:
            raise PipelineError("score shortlist contains an invalid candidate ID")
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
    analytical_input = output / "analytical-input.json"
    analytical_timing = output / "analytical-timing.json"
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
        [str(amoeba_opt), str(input_path), enumerate_option,
         f"--architecture-spec={architecture}", "-o", str(bound_ir)],
        "enumerate", output,
    )

    _run(
        [sys.executable, str(required_adapters["extract"]),
         "--input", str(bound_ir), "--output-dir", str(task_dir),
         "--index-output", str(task_index)],
        "extract-task-dfgs", output,
    )
    try:
        index = json.loads(task_index.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read extracted task DFG index: {error}") from error
    if not isinstance(index, dict):
        raise PipelineError("task DFG index is not a JSON object")
    task_arguments = _task_dfg_arguments(index)

    _run(
        [sys.executable, str(required_adapters["features"]),
         "--manifest", str(manifest), *task_arguments,
         "--neura-opt", str(amoeba_opt), "--architecture", str(architecture),
         "--output", str(analytical_input),
         "--timing-output", str(analytical_timing)],
        "analyze-task-shapes", output,
    )
    _run(
        [sys.executable, str(required_adapters["catalog"]),
         "--manifest", str(manifest),
         "--analytical-input", str(analytical_input), *task_arguments,
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
    if downstream and downstream[0] == "--":
        downstream.pop(0)
    shortlist = _read_shortlist(scores)
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
            "predicted_compute_bottleneck": item.get(
                "predicted_compute_bottleneck"
            ),
            "output": str(candidate_output),
        })

    report = {
        "input": str(input_path),
        "architecture": str(architecture),
        "predictor_submodule": str(PREDICTOR_ROOT),
        "top_k_requested": args.top_k,
        "real_pipeline_invocations": len(materialized),
        "downstream_arguments": downstream,
        "artifacts": {
            "candidate_manifest": str(manifest),
            "bound_ir": str(bound_ir),
            "analytical_input": str(analytical_input),
            "cost_catalog": str(cost_catalog),
            "scores": str(scores),
        },
        "shortlist": materialized,
        "todo": [
            "extend the candidate axes with fusion, fission, and tiling",
            "replace simultaneous packing with analytical spatial-temporal scheduling",
            "score communication and placement jointly with temporal order",
            "support symbolic task and allocation shapes after defining a finite domain",
        ],
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
    parser.add_argument("--max-cgras-per-task", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(raw_arguments)
    args.downstream_args = downstream_arguments
    if args.top_k <= 0:
        parser.error("--top-k must be positive for real-pipeline replay")
    if args.max_cgras_per_task <= 0 or args.max_candidates <= 0:
        parser.error("candidate limits must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, PipelineError, ValueError) as error:
        print(f"analytical task DSE failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
