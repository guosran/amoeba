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
    if kind == "cotiling":
        expected = {"kind", "producer_task", "consumer_task",
                    "producer_axis", "consumer_axis", "factor"}
        if set(raw) != expected:
            raise GraphSpaceError(f"{label} has invalid cotiling fields")
        producer = _checked_name(raw["producer_task"],
                                 f"{label}.producer_task")
        consumer = _checked_name(raw["consumer_task"],
                                 f"{label}.consumer_task")
        if producer == consumer:
            raise GraphSpaceError(
                f"{label} cannot cotile a task with itself")
        axes = []
        for field in ("producer_axis", "consumer_axis"):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GraphSpaceError(
                    f"{label}.{field} must be a non-negative integer")
            axes.append(value)
        return {"kind": kind, "producer_task": producer,
                "consumer_task": consumer, "producer_axis": axes[0],
                "consumer_axis": axes[1],
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
    completion_joins = raw.get("completion_joins", [])
    action_diagnostics = raw.get("action_diagnostics", [])
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
    if not isinstance(completion_joins, list):
        raise GraphSpaceError("graph facts completion_joins must be an array")
    if not isinstance(action_diagnostics, list):
        raise GraphSpaceError(
            "graph facts action_diagnostics must be an array")

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
        "completion_joins": completion_joins,
        "action_diagnostics": action_diagnostics,
        "structural_key": structural_key,
        "legal_actions": sorted(normalized_actions, key=_canonical_json),
    }


def _action_arguments(action: Mapping[str, Any]) -> List[str]:
    if action["kind"] == "fusion":
        mode = str(action["mode"]).replace("_", "-")
        return [("--fuse-task=first-task={} second-task={} "
                 "fusion-kind={}").format(
            action["first_task"], action["second_task"], mode)]
    if action["kind"] == "fission":
        return ["--fission-task=task-name={} left-nodes={}".format(
            action["task_name"],
            ":".join(str(value) for value in action["left_nodes"]))]
    if action["kind"] == "tiling":
        return ["--tile-task=task-name={} axis={} factor={}".format(
            action["task_name"], action["axis"], action["factor"])]
    if action["kind"] == "cotiling":
        return [
            "--tile-task=task-name={} axis={} factor={}".format(
                action["producer_task"], action["producer_axis"],
                action["factor"]),
            "--tile-task=task-name={} axis={} factor={}".format(
                action["consumer_task"], action["consumer_axis"],
                action["factor"]),
        ]
    raise GraphSpaceError(f"cannot execute action kind {action['kind']}")


def _selected_tile_factors(raw: object, min_factor: int,
                           max_factor: int) -> tuple[int, ...]:
    if raw is None:
        return tuple(
            factor for factor in (1, 2, 4, 8)
            if factor == 1 or min_factor <= factor <= max_factor
        )
    if raw == "":
        return tuple(range(min_factor, max_factor + 1))
    fields = raw.split(",") if isinstance(raw, str) else list(raw)
    try:
        factors = tuple(sorted({int(field) for field in fields}))
    except (TypeError, ValueError) as error:
        raise GraphSpaceError(
            "tile factors must be comma-separated integers"
        ) from error
    if (not factors or factors[0] < 1 or factors[-1] > max_factor or
            any(factor != 1 and factor < min_factor for factor in factors)):
        raise GraphSpaceError(
            "tile factors must contain 1 (identity) or values within the "
            "declared min/max transform bounds"
        )
    return factors


def _action_policy_rejection(
    action: Mapping[str, Any], min_tile_factor: int,
    selected_tile_factors: Sequence[int], forbid_nested_tiling: bool,
) -> Optional[str]:
    """Return the declared pruning reason for an otherwise legal action."""
    if action["kind"] not in ("tiling", "cotiling"):
        return None
    if action["factor"] < min_tile_factor:
        return "tiling_factor_below_minimum"
    if action["factor"] not in selected_tile_factors:
        return "tiling_factor_not_selected"
    # TileTaskPass gives every derived task a stable `.tile.` component. The
    # policy is intentionally syntactic and fail-closed: it only suppresses a
    # legal action when the optimizer has named its target as a prior tile.
    task_names = ([action["task_name"]] if action["kind"] == "tiling" else
                  [action["producer_task"], action["consumer_task"]])
    if forbid_nested_tiling and any(".tile." in name for name in task_names):
        return "nested_tiling"
    return None


def _action_step_cost(action: Mapping[str, Any], kind: str) -> int:
    """Returns how many per-transform steps one action consumes."""
    if action["kind"] == "cotiling":
        return 2 if kind == "tiling" else 0
    return 1 if action["kind"] == kind else 0


def _run(command: Sequence[str], label: str, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True,
                                   check=False,
                                   timeout=(timeout_seconds
                                            if timeout_seconds > 0 else None))
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
          *_action_arguments(action), "-o", str(output)],
         f"graph action {_canonical_json(action)}", timeout_seconds)
    if not output.is_file():
        raise GraphSpaceError("optimizer succeeded without writing rewritten IR")


def _run_witness(template: str, source: Path, candidate: Path,
                 variant_id: str, actions: Sequence[Mapping[str, Any]],
                 timeout_seconds: float) -> Dict[str, Any]:
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
        "{actions_json}": _canonical_json(list(actions)),
    }
    command = []
    for token in tokens:
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        command.append(token)
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False,
                                   timeout=(timeout_seconds
                                            if timeout_seconds > 0 else None))
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
        "actions_sha256": _sha256_bytes(
            _canonical_json(list(actions)).encode("utf-8")
        ),
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
    }


@dataclass
class _State:
    path_id: str
    parent_path_id: Optional[str]
    graph_variant_id: str
    parent_graph_variant_id: Optional[str]
    ir_path: Path
    facts: Dict[str, Any]
    structural_sha256: str
    actions: List[Dict[str, Any]]
    depth: int
    witness: Dict[str, Any]
    published: bool = True


def _structural_sha256(facts: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(facts["structural_key"]).encode("utf-8"))


def _variant_record(state: _State, artifact_name: str) -> Dict[str, Any]:
    ir_sha256 = _sha256_file(state.ir_path)
    return {
        "graph_variant_id": state.graph_variant_id,
        "graph_variant_sha256": ir_sha256,
        "lineage_path_id": state.path_id,
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
            "completion_joins": state.facts["completion_joins"],
            "action_diagnostics": state.facts["action_diagnostics"],
        },
        "shape_needs": state.facts["shape_needs"],
    }


def _traversal_record(state: _State) -> Dict[str, Any]:
    return {
        "path_id": state.path_id,
        "parent_path_id": state.parent_path_id,
        "graph_variant_id": state.graph_variant_id,
        "parent_graph_variant_id": state.parent_graph_variant_id,
        "depth": state.depth,
        "actions": state.actions,
        "structural_sha256": state.structural_sha256,
        "published": state.published,
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
            args.min_tile_factor < 2 or
            args.min_tile_factor > args.max_tile_factor or
            args.max_fission_actions_per_task <= 0 or
            args.max_duplicate_samples < 0 or
            args.max_fusion_steps < 0 or
            args.max_fission_steps < 0 or
            args.max_tiling_steps < 0 or
            args.timeout_seconds < 0):
        raise GraphSpaceError("invalid graph-space bound")
    if args.function:
        _checked_name(args.function, "function")
    if not args.witness_command:
        raise GraphSpaceError("a semantic witness command is required")
    if ("{source}" not in args.witness_command or
            "{candidate}" not in args.witness_command):
        raise GraphSpaceError(
            "witness command must reference {source} and {candidate}")
    selected_tile_factors = _selected_tile_factors(
        getattr(args, "tile_factors", ""), args.min_tile_factor,
        args.max_tile_factor,
    )
    fission_action_bound = (
        args.max_fission_actions_per_task
        if args.max_fission_steps > 0 else 0
    )

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
                                    fission_action_bound,
                                    args.timeout_seconds)
        if args.function and root_facts["function"] != args.function:
            raise GraphSpaceError("optimizer returned facts for another function")
        root_hash = _structural_sha256(root_facts)
        root_id = "graph-000000"
        root = _State(
            "path-000000", None, root_id, None, source_copy, root_facts,
            root_hash, [], 0,
            {"status": "verified", "scope": "byte-identical-source",
             "source_sha256": _sha256_file(source),
             "candidate_sha256": _sha256_file(source_copy)},
        )
        queue = deque([root])
        states = [root]
        traversal_states = [root]
        by_structure = {root_hash: root_id}
        # Publication is unique by structure, but traversal must retain the
        # non-dominated transform-budget states that can still expose
        # different legal descendants.
        budget_frontier = {root_hash: [(0, 0, 0)]}
        duplicates: List[Dict[str, Any]] = []
        duplicate_count = 0
        attempted_actions = 0
        pruned_actions = {"fusion": 0, "fission": 0, "tiling": 0}
        policy_pruned_actions = {
            "tiling_factor_below_minimum": 0,
            "tiling_factor_not_selected": 0,
            "nested_tiling": 0,
        }
        transform_limits = {
            "fusion": args.max_fusion_steps,
            "fission": args.max_fission_steps,
            "tiling": args.max_tiling_steps,
        }

        while queue:
            parent = queue.popleft()
            if parent.depth >= args.max_depth:
                if not parent.published:
                    parent.ir_path.unlink(missing_ok=True)
                continue
            for action in parent.facts["legal_actions"]:
                action_kind = action["kind"]
                budget_kind = ("tiling" if action_kind == "cotiling" else
                               action_kind)
                action_cost = _action_step_cost(action, budget_kind)
                used_steps = sum(
                    _action_step_cost(previous, budget_kind)
                    for previous in parent.actions
                )
                if used_steps + action_cost > transform_limits[budget_kind]:
                    pruned_actions[budget_kind] += 1
                    continue
                policy_rejection = _action_policy_rejection(
                    action, args.min_tile_factor, selected_tile_factors,
                    args.forbid_nested_tiling,
                )
                if policy_rejection is not None:
                    policy_pruned_actions[policy_rejection] += 1
                    continue
                attempted_actions += 1
                ordinal = attempted_actions
                candidate_id = f"graph-{ordinal:06d}"
                path_id = f"path-{ordinal:06d}"
                candidate_path = staging_root / f"candidate-{ordinal:06d}.mlir"
                _apply_action(optimizer, parent.ir_path, candidate_path, action,
                              args.timeout_seconds)
                candidate_facts_path = staging_root / f"facts-{ordinal:06d}.json"
                candidate_facts = _extract_facts(
                    optimizer, candidate_path, candidate_facts_path,
                    args.function, args.max_tile_factor,
                    fission_action_bound, args.timeout_seconds)
                candidate_facts_path.unlink(missing_ok=True)
                if candidate_facts["function"] != root_facts["function"]:
                    raise GraphSpaceError(
                        "rewritten graph facts changed the selected function")
                candidate_hash = _structural_sha256(candidate_facts)
                full_actions = parent.actions + [dict(action)]
                usage = tuple(sum(_action_step_cost(previous, kind)
                                  for previous in full_actions)
                              for kind in ("fusion", "fission", "tiling"))
                existing = by_structure.get(candidate_hash)
                if existing is not None:
                    duplicate_count += 1
                    frontier = budget_frontier[candidate_hash]
                    dominated = any(
                        all(saved <= current
                            for saved, current in zip(saved_usage, usage))
                        for saved_usage in frontier
                    )
                    retained_for_traversal = (
                        not dominated and parent.depth + 1 < args.max_depth
                    )
                    if retained_for_traversal:
                        frontier[:] = [
                            saved_usage for saved_usage in frontier
                            if not all(current <= saved
                                       for current, saved in zip(
                                           usage, saved_usage))
                        ]
                        frontier.append(usage)
                        duplicate_state = _State(
                            path_id, parent.path_id, existing,
                            parent.graph_variant_id, candidate_path,
                            candidate_facts, candidate_hash, full_actions,
                            parent.depth + 1,
                            {"status": "reused-from-structural-equivalent",
                             "graph_variant_id": existing},
                            published=False,
                        )
                        traversal_states.append(duplicate_state)
                        queue.append(duplicate_state)
                    else:
                        candidate_path.unlink(missing_ok=True)
                    if len(duplicates) < args.max_duplicate_samples:
                        duplicates.append({
                            "parent_graph_variant_id":
                                parent.graph_variant_id,
                            "parent_path_id": parent.path_id,
                            "path_id": (path_id
                                        if retained_for_traversal else None),
                            "actions": full_actions,
                            "duplicates_graph_variant_id": existing,
                            "structural_sha256": candidate_hash,
                            "deduplication": {
                                "policy": "canonical-structural-key-v1",
                                "semantic_witness_reused_from": existing,
                                "retained_for_budget_traversal":
                                    retained_for_traversal,
                                "transform_usage": {
                                    "fusion": usage[0],
                                    "fission": usage[1],
                                    "tiling": usage[2],
                                },
                            },
                        })
                    continue
                if len(states) >= args.max_variants:
                    raise GraphSpaceError(
                        "graph variant budget exceeded before complete traversal; "
                        "nothing was published")
                witness = _run_witness(
                    args.witness_command, source, candidate_path, candidate_id,
                    full_actions, args.timeout_seconds)
                state = _State(
                    path_id, parent.path_id, candidate_id,
                    parent.graph_variant_id, candidate_path, candidate_facts,
                    candidate_hash, full_actions, parent.depth + 1, witness,
                )
                by_structure[candidate_hash] = candidate_id
                budget_frontier[candidate_hash] = [usage]
                states.append(state)
                traversal_states.append(state)
                queue.append(state)
            if not parent.published:
                parent.ir_path.unlink(missing_ok=True)

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
                       "min_tile_factor": args.min_tile_factor,
                       "max_tile_factor": args.max_tile_factor,
                       "tile_factors": list(selected_tile_factors),
                       "tile_factor_one_is_identity": 1 in selected_tile_factors,
                       "forbid_nested_tiling": args.forbid_nested_tiling,
                       "max_fission_actions_per_task":
                           args.max_fission_actions_per_task,
                       "fission_fact_generation_enabled":
                           args.max_fission_steps > 0,
                       "max_duplicate_samples": args.max_duplicate_samples,
                       "max_fusion_steps": args.max_fusion_steps,
                       "max_fission_steps": args.max_fission_steps,
                       "max_tiling_steps": args.max_tiling_steps},
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
                       "traversal_states": len(traversal_states),
                       "duplicate_rewrite_sequences": duplicate_count,
                       "duplicate_provenance_samples": len(duplicates),
                       "duplicate_provenance_truncated":
                           duplicate_count > len(duplicates),
                       "attempted_actions": attempted_actions,
                       "semantic_witness_invocations": len(states) - 1,
                       "witnesses_avoided_by_structural_dedup": duplicate_count,
                       "pruned_actions_by_transform_limit": pruned_actions,
                       "pruned_actions_by_policy": policy_pruned_actions},
            "variants": records,
            "traversal_paths": [
                _traversal_record(state) for state in traversal_states
            ],
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
        for state in traversal_states:
            jsonl_records.append({
                "schema": MANIFEST_SCHEMA,
                "record_type": "traversal",
                **_traversal_record(state),
            })
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
    parser.add_argument("--min-tile-factor", type=int, default=2)
    parser.add_argument("--max-tile-factor", type=int, default=8)
    parser.add_argument(
        "--tile-factors", default=None,
        help=("Comma-separated selected factors. Factor 1 denotes the "
              "identity graph; defaults to 1,2,4,8 within the max bound."),
    )
    parser.add_argument(
        "--forbid-nested-tiling", action="store_true",
        help="Do not tile a task whose optimizer-assigned name is already a tile",
    )
    parser.add_argument("--max-fission-actions-per-task", type=int,
                        default=256)
    parser.add_argument("--max-duplicate-samples", type=int, default=100)
    parser.add_argument("--max-fusion-steps", type=int, default=2)
    parser.add_argument(
        "--max-fission-steps", type=int, default=0,
        help=("Maximum fission rewrites in one graph variant. Fission is "
              "temporarily disabled by default; pass a positive value to "
              "opt in."),
    )
    parser.add_argument("--max-tiling-steps", type=int, default=2)
    parser.add_argument("--function", default="",
                        help="Exact Taskflow function to enumerate")
    parser.add_argument("--witness-command", required=True,
                        help="Checked command with {source}, {candidate}, and "
                             "{actions_json}, plus optional "
                             "{graph_variant_id}, placeholders")
    parser.add_argument(
        "--timeout-seconds", type=float, default=0.0,
        help="Bound each optimizer/witness call; zero means unbounded",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The joint source-bound rewrite space is an opt-in MLIR pass mode.
    # The pass owns legality and graph identity without entering the backend
    # shape, mapping, or cost exploration path.
    # The semantic GEMM axis is a source-derived task graph space. It shares
    # this program-level entry but does not enter the shape/replica rewrite BFS.
    # The imported module owns the canonical M/N/K rules and IDs.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--semantic-joint-tiling-fusion" in arguments:
        from orbit_joint_mlir_pass_entry import main as joint_main
        arguments.remove("--semantic-joint-tiling-fusion")
        return joint_main(arguments)
    if "--semantic-gemm-task-variants" in arguments:
        from orbit_semantic_graph_candidates import main as semantic_main
        arguments.remove("--semantic-gemm-task-variants")
        return semantic_main(arguments)
    try:
        manifest = enumerate_graph_space(parse_args(arguments))
    except (GraphSpaceError, OSError) as error:
        print(f"joint graph enumeration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
