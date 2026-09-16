#!/usr/bin/env python3
"""Enumerate bounded, witnessed Taskflow graph rewrite sequences.

The optimizer is the authority for legal actions.  After every successful
rewrite this driver asks it for fresh graph facts, so an action is never
replayed against stale topology.  No transformed graph is published until
its semantic witness succeeds and the complete bounded traversal finishes.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FACT_SCHEMA = "amoeba-joint-task-graph-facts-v1"
MANIFEST_SCHEMA = "amoeba-joint-task-graph-variants-v1"
PREPARATION_CONTRACT = "amoeba-graph-variant-preparation-v1"
PREPARATION_PASSES = [
    "--classify-task-and-counter",
    "--convert-taskflow-to-neura",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--convert-cf-to-llvm",
    "--assign-accelerator",
    "--lower-memref-to-neura",
    "--lower-arith-to-neura",
    "--lower-builtin-to-neura",
    "--lower-llvm-to-neura",
    "--promote-input-arg-to-const",
    "--fold-constant",
    "--canonicalize-return",
    "--canonicalize-live-in",
    "--leverage-predicated-value",
    "--transform-ctrl-to-data-flow",
    "--fold-constant",
]
DATA_MOVEMENT_PASS = "--insert-data-mov"


class GraphSpaceError(RuntimeError):
    """A graph-space contract failed before transactional publication."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_name(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value or "\x00" in value or
            any(character.isspace() for character in value) or "=" in value):
        raise GraphSpaceError(f"{label} must be a non-empty option-safe name")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphSpaceError(f"{label} must be a positive integer")
    return value


def _validate_action(raw: Any, label: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise GraphSpaceError(f"{label} must be an object")
    kind = raw.get("kind")
    if kind == "fusion":
        expected = {"kind", "mode", "first_task", "second_task"}
        if set(raw) != expected:
            raise GraphSpaceError(f"{label} has invalid fusion fields")
        mode = raw["mode"]
        if mode not in ("producer_consumer", "sibling"):
            raise GraphSpaceError(f"{label}.mode is invalid")
        first = _checked_name(raw["first_task"], f"{label}.first_task")
        second = _checked_name(raw["second_task"], f"{label}.second_task")
        if first == second:
            raise GraphSpaceError(f"{label} cannot fuse a task with itself")
        return {"kind": kind, "mode": mode, "first_task": first,
                "second_task": second}
    if kind == "fission":
        expected = {"kind", "task_name", "left_nodes"}
        if set(raw) != expected:
            raise GraphSpaceError(f"{label} has invalid fission fields")
        left_nodes = raw["left_nodes"]
        if (not isinstance(left_nodes, list) or not left_nodes or
                any(isinstance(value, bool) or not isinstance(value, int) or
                    value < 0 for value in left_nodes) or
                left_nodes != sorted(set(left_nodes))):
            raise GraphSpaceError(
                f"{label}.left_nodes must be unique increasing ordinals")
        return {"kind": kind,
                "task_name": _checked_name(raw["task_name"],
                                             f"{label}.task_name"),
                "left_nodes": left_nodes}
    if kind == "tiling":
        expected = {"kind", "task_name", "axis", "factor"}
        if set(raw) != expected:
            raise GraphSpaceError(f"{label} has invalid tiling fields")
        axis = raw["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise GraphSpaceError(f"{label}.axis must be a non-negative integer")
        return {"kind": kind,
                "task_name": _checked_name(raw["task_name"],
                                             f"{label}.task_name"),
                "axis": axis,
                "factor": _positive_integer(raw["factor"],
                                             f"{label}.factor")}
    raise GraphSpaceError(f"{label}.kind is unsupported")


def _validate_facts(raw: Any, source: Path) -> Dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != FACT_SCHEMA:
        raise GraphSpaceError(f"invalid graph facts emitted for {source}")
    function = _checked_name(raw.get("function"), "facts.function")
    tasks = raw.get("tasks")
    dependencies = raw.get("dependencies")
    shape_needs = raw.get("shape_needs")
    structural_key = raw.get("structural_key")
    actions = raw.get("legal_actions")
    if not isinstance(tasks, list) or not tasks:
        raise GraphSpaceError("graph facts must contain at least one task")
    if not isinstance(dependencies, list):
        raise GraphSpaceError("graph facts dependencies must be an array")
    if not isinstance(shape_needs, list):
        raise GraphSpaceError("graph facts shape_needs must be an array")
    if not isinstance(structural_key, dict):
        raise GraphSpaceError("graph facts structural_key must be an object")
    if not isinstance(actions, list):
        raise GraphSpaceError("graph facts legal_actions must be an array")

    names: set[str] = set()
    normalized_tasks: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise GraphSpaceError(f"facts.tasks[{index}] must be an object")
        name = _checked_name(task.get("task_name"),
                             f"facts.tasks[{index}].task_name")
        if name in names:
            raise GraphSpaceError(f"duplicate task name in graph facts: {name}")
        names.add(name)
        body_hash = task.get("body_sha256")
        if (not isinstance(body_hash, str) or len(body_hash) != 64 or
                any(c not in "0123456789abcdef" for c in body_hash)):
            raise GraphSpaceError(f"facts.tasks[{index}].body_sha256 is invalid")
        normalized_tasks.append(dict(task))

    normalized_dependencies: List[Dict[str, Any]] = []
    for index, edge in enumerate(dependencies):
        if not isinstance(edge, dict):
            raise GraphSpaceError(
                f"facts.dependencies[{index}] must be an object")
        source_name = _checked_name(edge.get("source_task"),
                                    f"facts.dependencies[{index}].source_task")
        target_name = _checked_name(edge.get("target_task"),
                                    f"facts.dependencies[{index}].target_task")
        if source_name not in names or target_name not in names:
            raise GraphSpaceError("graph dependency names an unknown task")
        if edge.get("kind") not in ("raw", "war", "waw", "value", "control"):
            raise GraphSpaceError("graph dependency has an unknown kind")
        normalized_dependencies.append(dict(edge))

    normalized_actions = [
        _validate_action(action, f"facts.legal_actions[{index}]")
        for index, action in enumerate(actions)
    ]
    if len({_canonical_json(action) for action in normalized_actions}) != len(
            normalized_actions):
        raise GraphSpaceError("graph facts contain duplicate legal actions")
    return {
        "schema": FACT_SCHEMA,
        "function": function,
        "tasks": normalized_tasks,
        "dependencies": normalized_dependencies,
        "shape_needs": shape_needs,
        "structural_key": structural_key,
        "legal_actions": sorted(normalized_actions, key=_canonical_json),
    }


def _action_argument(action: Mapping[str, Any]) -> str:
    if action["kind"] == "fusion":
        mode = str(action["mode"]).replace("_", "-")
        return ("--fuse-task=first-task={} second-task={} fusion-kind={}".format(
            action["first_task"], action["second_task"], mode))
    if action["kind"] == "fission":
        return "--fission-task=task-name={} left-nodes={}".format(
            action["task_name"],
            ":".join(str(value) for value in action["left_nodes"]))
    if action["kind"] == "tiling":
        return "--tile-task=task-name={} axis={} factor={}".format(
            action["task_name"], action["axis"], action["factor"])
    raise GraphSpaceError(f"cannot execute action kind {action['kind']}")


def _run(command: Sequence[str], label: str, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True,
                                   check=False, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GraphSpaceError(f"{label} could not run: {error}") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise GraphSpaceError(
            f"{label} failed with exit code {completed.returncode}: {detail}")


def _extract_facts(optimizer: Path, source: Path, output: Path,
                   function_name: str, max_tile_factor: int,
                   max_fission_actions_per_task: int,
                   timeout_seconds: float) -> Dict[str, Any]:
    function_option = f" function={function_name}" if function_name else ""
    command = [
        str(optimizer), str(source), "--verify-each",
        ("--extract-joint-task-graph-facts=output={} max-tile-factor={} "
         "max-fission-actions-per-task={}{}").format(
            output, max_tile_factor, max_fission_actions_per_task,
            function_option),
        "-o", os.devnull,
    ]
    _run(command, f"graph fact extraction for {source.name}", timeout_seconds)
    try:
        raw = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphSpaceError(f"cannot read graph facts {output}: {error}") from error
    return _validate_facts(raw, source)


def _apply_action(optimizer: Path, source: Path, output: Path,
                  action: Mapping[str, Any], timeout_seconds: float) -> None:
    _run([str(optimizer), str(source), "--verify-each",
          _action_argument(action), "-o", str(output)],
         f"graph action {_canonical_json(action)}", timeout_seconds)
    if not output.is_file():
        raise GraphSpaceError("optimizer succeeded without writing rewritten IR")


def _run_witness(template: str, source: Path, candidate: Path,
                 variant_id: str, timeout_seconds: float) -> Dict[str, Any]:
    try:
        tokens = shlex.split(template)
    except ValueError as error:
        raise GraphSpaceError(f"invalid witness command: {error}") from error
    if not tokens:
        raise GraphSpaceError("witness command must not be empty")
    replacements = {
        "{source}": str(source),
        "{candidate}": str(candidate),
        "{graph_variant_id}": variant_id,
    }
    command = []
    for token in tokens:
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        command.append(token)
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GraphSpaceError(
            f"semantic witness failed closed for {variant_id}: {error}") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise GraphSpaceError(
            "semantic witness failed closed for {} with exit code {}: {}".format(
                variant_id, completed.returncode, detail))
    return {
        "status": "verified",
        "command_template": template,
        "argv_template": tokens,
        "graph_variant_id": variant_id,
        "source_sha256": _sha256_file(source),
        "candidate_sha256": _sha256_file(candidate),
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
    }


@dataclass
class _State:
    graph_variant_id: str
    parent_graph_variant_id: Optional[str]
    ir_path: Path
    facts: Dict[str, Any]
    structural_sha256: str
    actions: List[Dict[str, Any]]
    depth: int
    witness: Dict[str, Any]


def _structural_sha256(facts: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(facts["structural_key"]).encode("utf-8"))


def _variant_record(state: _State, artifact_name: str) -> Dict[str, Any]:
    ir_sha256 = _sha256_file(state.ir_path)
    return {
        "graph_variant_id": state.graph_variant_id,
        "graph_variant_sha256": ir_sha256,
        "parent_graph_variant_id": state.parent_graph_variant_id,
        "depth": state.depth,
        "actions": state.actions,
        "structural_sha256": state.structural_sha256,
        "result_ir": {"artifact": artifact_name,
                      "path": artifact_name,
                      "path_base": "artifact_dir",
                      "sha256": ir_sha256},
        "semantic_witness": state.witness,
        "task_graph": {
            "function": state.facts["function"],
            "tasks": state.facts["tasks"],
            "dependencies": state.facts["dependencies"],
        },
        "shape_needs": state.facts["shape_needs"],
    }


def enumerate_graph_space(args: argparse.Namespace) -> Dict[str, Any]:
    source = args.input.expanduser().resolve()
    optimizer = args.amoeba_opt.expanduser().resolve()
    output = args.output.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    artifact_dir_from_manifest = os.path.relpath(artifact_dir, output.parent)
    if not source.is_file():
        raise GraphSpaceError(f"input MLIR does not exist: {source}")
    if not optimizer.is_file() or not os.access(optimizer, os.X_OK):
        raise GraphSpaceError(f"optimizer is not executable: {optimizer}")
    if output.exists() or artifact_dir.exists():
        raise GraphSpaceError("output and artifact directory must not exist")
    if (args.max_depth < 0 or args.max_variants <= 0 or
            args.max_tile_factor < 2 or
            args.max_fission_actions_per_task <= 0 or
            args.timeout_seconds <= 0):
        raise GraphSpaceError("invalid graph-space bound")
    if args.function:
        _checked_name(args.function, "function")
    if not args.witness_command:
        raise GraphSpaceError("a semantic witness command is required")
    if ("{source}" not in args.witness_command or
            "{candidate}" not in args.witness_command):
        raise GraphSpaceError(
            "witness command must reference {source} and {candidate}")

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=".joint-graph-variants-", dir=artifact_dir.parent))
    staged_artifacts = staging_root / "artifacts"
    staged_artifacts.mkdir()
    staged_manifest = staging_root / "manifest.json"
    published_artifacts = False
    try:
        source_copy = staging_root / "source.mlir"
        shutil.copyfile(source, source_copy)
        facts_path = staging_root / "facts-000000.json"
        root_facts = _extract_facts(optimizer, source_copy, facts_path,
                                    args.function,
                                    args.max_tile_factor,
                                    args.max_fission_actions_per_task,
                                    args.timeout_seconds)
        if args.function and root_facts["function"] != args.function:
            raise GraphSpaceError("optimizer returned facts for another function")
        root_hash = _structural_sha256(root_facts)
        root_id = "graph-000000"
        root = _State(
            root_id, None, source_copy, root_facts, root_hash, [], 0,
            {"status": "verified", "scope": "byte-identical-source",
             "source_sha256": _sha256_file(source),
             "candidate_sha256": _sha256_file(source_copy)},
        )
        queue = deque([root])
        states = [root]
        by_structure = {root_hash: root_id}
        duplicates: List[Dict[str, Any]] = []
        attempted_actions = 0

        while queue:
            parent = queue.popleft()
            if parent.depth >= args.max_depth:
                continue
            for action in parent.facts["legal_actions"]:
                attempted_actions += 1
                ordinal = attempted_actions
                candidate_id = f"graph-{ordinal:06d}"
                candidate_path = staging_root / f"candidate-{ordinal:06d}.mlir"
                _apply_action(optimizer, parent.ir_path, candidate_path, action,
                              args.timeout_seconds)
                candidate_facts_path = staging_root / f"facts-{ordinal:06d}.json"
                candidate_facts = _extract_facts(
                    optimizer, candidate_path, candidate_facts_path,
                    args.function, args.max_tile_factor,
                    args.max_fission_actions_per_task, args.timeout_seconds)
                if candidate_facts["function"] != root_facts["function"]:
                    raise GraphSpaceError(
                        "rewritten graph facts changed the selected function")
                candidate_hash = _structural_sha256(candidate_facts)
                witness = _run_witness(
                    args.witness_command, source, candidate_path, candidate_id,
                    args.timeout_seconds)
                full_actions = parent.actions + [dict(action)]
                existing = by_structure.get(candidate_hash)
                if existing is not None:
                    duplicates.append({
                        "parent_graph_variant_id": parent.graph_variant_id,
                        "actions": full_actions,
                        "duplicates_graph_variant_id": existing,
                        "structural_sha256": candidate_hash,
                        "semantic_witness": witness,
                    })
                    continue
                if len(states) >= args.max_variants:
                    raise GraphSpaceError(
                        "graph variant budget exceeded before complete traversal; "
                        "nothing was published")
                state = _State(
                    candidate_id, parent.graph_variant_id, candidate_path,
                    candidate_facts, candidate_hash, full_actions,
                    parent.depth + 1, witness,
                )
                by_structure[candidate_hash] = candidate_id
                states.append(state)
                queue.append(state)

        records = []
        for state in states:
            artifact_name = f"{state.graph_variant_id}.mlir"
            shutil.copyfile(state.ir_path, staged_artifacts / artifact_name)
            records.append(_variant_record(state, artifact_name))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "complete": True,
            "source": {"path": str(source), "sha256": _sha256_file(source)},
            "optimizer": {"path": str(optimizer),
                          "sha256": _sha256_file(optimizer)},
            "bounds": {"max_depth": args.max_depth,
                       "max_variants": args.max_variants,
                       "max_tile_factor": args.max_tile_factor,
                       "max_fission_actions_per_task":
                           args.max_fission_actions_per_task},
            "preparation": {
                "contract": PREPARATION_CONTRACT,
                "optimizer_args": PREPARATION_PASSES,
                "data_movement": {
                    "optimizer_arg": DATA_MOVEMENT_PASS,
                    "required_exactly_once": True,
                    "position": "after graph preparation and before feature extraction",
                },
                "output_contract": "one-pre-mapper-neura-kernel-per-compute-task",
            },
            "function": root_facts["function"],
            "counts": {"unique_graph_variants": len(states),
                       "duplicate_rewrite_sequences": len(duplicates),
                       "attempted_actions": attempted_actions},
            "variants": records,
            "duplicates": duplicates,
        }
        header = {
            "schema": MANIFEST_SCHEMA,
            "record_type": "header",
            "source": manifest["source"],
            "optimizer": manifest["optimizer"],
            "bounds": manifest["bounds"],
            "function": manifest["function"],
            "artifact_dir": {"path": artifact_dir_from_manifest,
                             "path_base": "manifest_dir",
                             "variant_paths_relative_to": "this-directory"},
            "preparation": manifest["preparation"],
            "identity_included": True,
            "publication_policy": "unique-and-witness-verified-only",
        }
        jsonl_records: List[Dict[str, Any]] = [header]
        for record in records:
            jsonl_records.append({"schema": MANIFEST_SCHEMA,
                                  "record_type": "variant", **record})
        for duplicate in duplicates:
            jsonl_records.append({"schema": MANIFEST_SCHEMA,
                                  "record_type": "duplicate", **duplicate})
        jsonl_records.append({
            "schema": MANIFEST_SCHEMA,
            "record_type": "footer",
            "complete": True,
            **manifest["counts"],
        })
        staged_manifest.write_text(
            "".join(_canonical_json(record) + "\n"
                    for record in jsonl_records), encoding="utf-8")

        os.replace(staged_artifacts, artifact_dir)
        published_artifacts = True
        try:
            os.replace(staged_manifest, output)
        except BaseException:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            published_artifacts = False
            raise
        return manifest
    finally:
        if not published_artifacts and artifact_dir.exists() and not output.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        shutil.rmtree(staging_root, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-variants", type=int, default=1000)
    parser.add_argument("--max-tile-factor", type=int, default=8)
    parser.add_argument("--max-fission-actions-per-task", type=int,
                        default=256)
    parser.add_argument("--function", default="",
                        help="Exact Taskflow function to enumerate")
    parser.add_argument("--witness-command", required=True,
                        help="Checked command with {source}, {candidate}, and "
                             "optional {graph_variant_id} placeholders")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        manifest = enumerate_graph_space(parse_args(argv))
    except (GraphSpaceError, OSError) as error:
        print(f"joint graph enumeration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
