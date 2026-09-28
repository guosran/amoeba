#!/usr/bin/env python3
"""Validate and rank complete graph-variant candidate score spaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


GRAPH_SCHEMA = "amoeba-joint-task-graph-variants-v1"
GRAPH_PREPARATION_CONTRACT = "amoeba-graph-variant-preparation-v1"
CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates"
COST_SCHEMA = "amoeba-task-shape-cost"
SCORE_SCHEMA = "amoeba-analytical-task-scores"
GLOBAL_SCHEMA = "amoeba-joint-program-ranking-v1"
PLACEMENT_POLICY_RECORD = {
    "exact": "global-fixed-order-exact-v2",
    "heuristic": "dependency-ready-greedy-v1",
}
ACTIVITY_POLICY_RECORD = {
    "fixed": "fixed-task-priority-v1",
    "critical-path": "weighted-critical-path-ready-list-v1",
    "pipeline-aware": "makespan-bound-pipeline-tiebreak-v1",
    "beam": "bounded-activity-placement-beam-v1",
    "exact": "exact-activity-placement-oracle-v1",
}
SCHEDULE_TRACE_SCHEMA = "joint-schedule-trace-v3"


class RankingError(RuntimeError):
    """A provenance or completeness contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RankingError(f"{label} must be an object")
    return value


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RankingError(f"{label} is not a lowercase SHA-256")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _json_sha256(value: object) -> str:
    try:
        payload = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RankingError("semantic witness actions are not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _records(path: Path, schema: str) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RankingError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RankingError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        record = _object(record, f"record at {path}:{line_number}")
        if record.get("schema") != schema:
            raise RankingError(f"schema mismatch at {path}:{line_number}")
        records.append(record)
    if not records:
        raise RankingError(f"empty JSONL artifact: {path}")
    return records


def _resolved_artifact(manifest: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RankingError("artifact path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


@dataclass(frozen=True)
class GraphVariant:
    index: int
    graph_variant_id: str
    ir_path: Path
    ir_sha256: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class VariantArtifacts:
    graph: GraphVariant
    candidates: Path
    costs: Path
    scores: Path


def load_graph_variants(path: Path) -> Tuple[Mapping[str, Any], List[GraphVariant]]:
    records = _records(path, GRAPH_SCHEMA)
    if records[0].get("record_type") != "header":
        raise RankingError("graph manifest must start with one header")
    if records[-1].get("record_type") != "footer":
        raise RankingError("graph manifest must end with one footer")
    if any(record.get("record_type") == "header" for record in records[1:]):
        raise RankingError("graph manifest contains a misplaced header")
    if any(record.get("record_type") == "footer" for record in records[:-1]):
        raise RankingError("graph manifest contains a misplaced footer")
    header = records[0]
    footer = records[-1]
    if footer.get("complete") is not True:
        raise RankingError("graph manifest is not complete")

    artifact_directory = _object(
        header.get("artifact_dir"), "graph artifact directory"
    )
    if artifact_directory.get("path_base") != "manifest_dir":
        raise RankingError("graph artifact directory base is unsupported")
    artifact_root = _resolved_artifact(path, artifact_directory.get("path"))
    if not artifact_root.is_dir():
        raise RankingError("graph artifact directory does not exist")
    preparation = _object(header.get("preparation"), "graph preparation")
    if preparation.get("contract") != GRAPH_PREPARATION_CONTRACT:
        raise RankingError("graph preparation contract is unsupported")

    source = _object(header.get("source"), "graph source")
    source_path = _resolved_artifact(path, source.get("path"))
    source_sha = _sha(source.get("sha256"), "graph source SHA-256")
    if not source_path.is_file() or sha256_file(source_path) != source_sha:
        raise RankingError("graph source byte identity mismatch")

    traversal_by_id: Dict[str, Mapping[str, Any]] = {}
    saw_root_path = False
    for record in records[1:-1]:
        if record.get("record_type") != "traversal":
            continue
        path_id = record.get("path_id")
        if (
            not isinstance(path_id, str)
            or not path_id
            or path_id in traversal_by_id
        ):
            raise RankingError("graph traversal path identity is invalid")
        graph_id = record.get("graph_variant_id")
        if not isinstance(graph_id, str) or not graph_id:
            raise RankingError(f"graph traversal {path_id} has no graph identity")
        actions = record.get("actions")
        depth = record.get("depth")
        if (
            not isinstance(actions, list)
            or any(not isinstance(action, Mapping) for action in actions)
            or isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth != len(actions)
            or not isinstance(record.get("published"), bool)
        ):
            raise RankingError(f"graph traversal {path_id} is malformed")
        _sha(
            record.get("structural_sha256"),
            f"graph traversal {path_id} structural SHA-256",
        )
        parent_path_id = record.get("parent_path_id")
        parent_graph_id = record.get("parent_graph_variant_id")
        if parent_path_id is None:
            if saw_root_path or actions or depth != 0 or parent_graph_id is not None:
                raise RankingError("graph traversal has an invalid root path")
            saw_root_path = True
        else:
            if (
                not isinstance(parent_path_id, str)
                or parent_path_id not in traversal_by_id
            ):
                raise RankingError(
                    f"graph traversal {path_id} has an invalid parent path"
                )
            parent_path = traversal_by_id[parent_path_id]
            parent_actions = parent_path["actions"]
            if (
                actions[:-1] != parent_actions
                or depth != parent_path["depth"] + 1
                or parent_graph_id != parent_path["graph_variant_id"]
            ):
                raise RankingError(
                    f"graph traversal {path_id} does not extend its parent"
                )
        traversal_by_id[path_id] = record
    if not traversal_by_id or not saw_root_path:
        raise RankingError("graph manifest has no traversal lineage")

    variants: List[GraphVariant] = []
    duplicates = 0
    identifiers = set()
    structures = set()
    referenced_published_paths = set()
    saw_identity = False
    for record in records[1:-1]:
        kind = record.get("record_type")
        if kind == "traversal":
            continue
        if kind == "duplicate":
            duplicates += 1
            continue
        if kind != "variant":
            raise RankingError("graph manifest has an unknown record type")
        identifier = record.get("graph_variant_id")
        if not isinstance(identifier, str) or not identifier:
            raise RankingError("graph variant identity is missing")
        if identifier in identifiers:
            raise RankingError("graph variant identity is duplicated")
        identifiers.add(identifier)
        structure = _sha(
            record.get("structural_sha256"), "graph structural SHA-256"
        )
        if structure in structures:
            raise RankingError("unique graph records duplicate a structure")
        structures.add(structure)
        result = _object(record.get("result_ir"), "graph result IR")
        if result.get("path_base") != "artifact_dir":
            raise RankingError("graph result IR path base is unsupported")
        raw_ir_path = result.get("path")
        if not isinstance(raw_ir_path, str) or not raw_ir_path:
            raise RankingError("graph result IR path is missing")
        ir_path = (artifact_root / raw_ir_path).resolve()
        try:
            ir_path.relative_to(artifact_root)
        except ValueError as error:
            raise RankingError(
                "graph result IR escapes its artifact directory"
            ) from error
        ir_sha = _sha(result.get("sha256"), "graph result IR SHA-256")
        if record.get("graph_variant_sha256") != ir_sha:
            raise RankingError("graph variant SHA-256 fields disagree")
        if not ir_path.is_file() or sha256_file(ir_path) != ir_sha:
            raise RankingError(f"graph result IR mismatch for {identifier}")

        actions = record.get("actions")
        depth = record.get("depth")
        if not isinstance(actions, list) or any(
            not isinstance(action, Mapping) for action in actions
        ) or isinstance(depth, bool) or not isinstance(depth, int) or depth != len(
            actions
        ):
            raise RankingError(f"graph variant {identifier} actions are invalid")
        lineage_path_id = record.get("lineage_path_id")
        if (
            not isinstance(lineage_path_id, str)
            or lineage_path_id not in traversal_by_id
            or lineage_path_id in referenced_published_paths
        ):
            raise RankingError(
                f"graph variant {identifier} has invalid traversal lineage"
            )
        lineage = traversal_by_id[lineage_path_id]
        parent_path_id = lineage.get("parent_path_id")
        expected_parent = (
            None
            if parent_path_id is None
            else traversal_by_id[str(parent_path_id)]["graph_variant_id"]
        )
        if (
            lineage.get("published") is not True
            or lineage.get("graph_variant_id") != identifier
            or lineage.get("structural_sha256") != structure
            or lineage.get("actions") != actions
            or lineage.get("depth") != depth
            or record.get("parent_graph_variant_id") != expected_parent
        ):
            raise RankingError(
                f"graph variant {identifier} disagrees with traversal lineage"
            )
        referenced_published_paths.add(lineage_path_id)
        witness = _object(record.get("semantic_witness"), "semantic witness")
        if witness.get("status") != "verified":
            raise RankingError(f"graph variant {identifier} is not witnessed")
        witness_source_sha = _sha(
            witness.get("source_sha256"), "semantic witness source SHA-256"
        )
        witness_candidate_sha = _sha(
            witness.get("candidate_sha256"),
            "semantic witness candidate SHA-256",
        )
        if witness_source_sha != source_sha or witness_candidate_sha != ir_sha:
            raise RankingError(
                f"graph variant {identifier} semantic witness byte identity "
                "mismatch"
            )

        parent = record.get("parent_graph_variant_id")
        if parent is None:
            if (
                saw_identity
                or variants
                or actions
                or witness.get("scope") != "byte-identical-source"
                or ir_sha != source_sha
            ):
                raise RankingError(
                    f"graph variant {identifier} has an invalid identity witness"
                )
            saw_identity = True
        else:
            if (
                not isinstance(parent, str)
                or not parent
                or parent == identifier
                or parent not in identifiers
                or not actions
            ):
                raise RankingError(
                    f"graph variant {identifier} has invalid rewrite provenance"
                )
            command_template = witness.get("command_template")
            if not isinstance(command_template, str) or not command_template:
                raise RankingError(
                    f"graph variant {identifier} is not witnessed"
                )
            if witness.get("graph_variant_id") != identifier:
                raise RankingError(
                    f"graph variant {identifier} semantic witness identity "
                    "mismatch"
                )
            witness_actions_sha = _sha(
                witness.get("actions_sha256"),
                "semantic witness actions SHA-256",
            )
            if witness_actions_sha != _json_sha256(actions):
                raise RankingError(
                    f"graph variant {identifier} semantic witness action "
                    "mismatch"
                )
        variants.append(
            GraphVariant(len(variants), identifier, ir_path, ir_sha, record)
        )

    if not variants:
        raise RankingError("graph manifest contains no unique variant")
    if any(
        path.get("graph_variant_id") not in identifiers
        for path in traversal_by_id.values()
    ):
        raise RankingError("graph traversal references an unknown canonical graph")
    published_paths = {
        path_id
        for path_id, record in traversal_by_id.items()
        if record.get("published") is True
    }
    if published_paths != referenced_published_paths:
        raise RankingError("graph traversal publication set is inconsistent")
    if footer.get("unique_graph_variants") != len(variants):
        raise RankingError("graph manifest unique count is inconsistent")
    if footer.get("traversal_states") != len(traversal_by_id):
        raise RankingError("graph manifest traversal count is inconsistent")
    duplicate_total = footer.get("duplicate_rewrite_sequences")
    duplicate_samples = footer.get("duplicate_provenance_samples")
    if duplicate_samples is None:
        duplicate_samples = duplicate_total
    if (not isinstance(duplicate_total, int) or duplicate_total < duplicates or
            duplicate_samples != duplicates or
            footer.get("duplicate_provenance_truncated", False)
            != (duplicate_total > duplicates)):
        raise RankingError("graph manifest duplicate count is inconsistent")
    validated_header = dict(header)
    validated_header["enumeration_counts"] = {
        key: value for key, value in footer.items()
        if key not in {"schema", "record_type", "complete"}
    }
    return validated_header, variants


def _load_candidates(
    path: Path, graph: GraphVariant
) -> Tuple[Mapping[str, Any], List[str]]:
    records = _records(path, CANDIDATE_SCHEMA)
    if records[0].get("record_type") != "header" or records[-1].get(
        "record_type"
    ) != "footer":
        raise RankingError("candidate manifest is missing header/footer")
    header = records[0]
    if (
        header.get("graph_variant_id") != graph.graph_variant_id
        or header.get("graph_variant_sha256") != graph.ir_sha256
    ):
        raise RankingError("candidate manifest does not match its graph variant")
    candidates: List[str] = []
    for record in records[1:-1]:
        if record.get("record_type") != "candidate":
            raise RankingError("candidate manifest has a misplaced record")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidates:
            raise RankingError("candidate identity is invalid or duplicated")
        candidates.append(candidate_id)
    if records[-1].get("candidate_count") != len(candidates):
        raise RankingError("candidate manifest is incomplete")
    return header, candidates


def _load_cost(path: Path) -> Mapping[str, Any]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RankingError(f"cannot read cost catalogue {path}: {error}") from error
    root = _object(root, "cost catalogue")
    if root.get("schema") != COST_SCHEMA:
        raise RankingError("cost catalogue schema mismatch")
    if not isinstance(root.get("namespace"), str) or not root["namespace"]:
        raise RankingError("cost catalogue namespace is missing")
    return root


def _load_scores(
    path: Path,
    candidates_path: Path,
    costs_path: Path,
    graph: GraphVariant,
    architecture_sha256: str,
    communication_mode: str,
    required_top_k: int,
    placement_search_policy: str,
    activity_scheduling_policy: str,
    activity_beam_width: int,
) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]], int]:
    candidate_header, candidate_ids = _load_candidates(candidates_path, graph)
    costs = _load_cost(costs_path)
    records = _records(path, SCORE_SCHEMA)
    if records[0].get("record_type") != "header" or records[-1].get(
        "record_type"
    ) != "footer":
        raise RankingError("score manifest is missing header/footer")
    header = records[0]
    footer = records[-1]
    candidate_architecture = _object(
        candidate_header.get("architecture"), "candidate architecture"
    )
    if (
        header.get("candidate_schema") != CANDIDATE_SCHEMA
        or header.get("candidate_manifest_sha256") != sha256_file(candidates_path)
        or header.get("cost_catalog_sha256") != sha256_file(costs_path)
        or header.get("cost_namespace") != costs.get("namespace")
        or header.get("architecture_sha256") != architecture_sha256
        or candidate_architecture.get("spec_sha256") != architecture_sha256
        or header.get("communication_mode") != communication_mode
        or header.get("schedule_trace_schema") != SCHEDULE_TRACE_SCHEMA
        or header.get("placement_search_policy")
        != PLACEMENT_POLICY_RECORD[placement_search_policy]
        or header.get("activity_scheduling_policy")
        != ACTIVITY_POLICY_RECORD[activity_scheduling_policy]
        or header.get("activity_beam_width") != activity_beam_width
        or header.get("function") != candidate_header.get("function")
        or header.get("graph_variant_id") != graph.graph_variant_id
        or header.get("graph_variant_sha256") != graph.ir_sha256
    ):
        raise RankingError("score provenance does not match variant artifacts")
    scores: List[Mapping[str, Any]] = []
    score_ids = set()
    for record in records[1:-1]:
        if record.get("record_type") != "score":
            raise RankingError("score manifest has a misplaced record")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in score_ids:
            raise RankingError("score identity is invalid or duplicated")
        score_ids.add(candidate_id)
        scores.append(record)
    candidate_id_set = set(candidate_ids)
    output_scope = header.get("output_scope", "all-candidates")
    valid_count = sum(record.get("valid") is True for record in scores)
    evaluated_valid_count = footer.get("evaluated_valid_count", valid_count)
    common_counts_valid = (
        footer.get("candidate_count") == len(candidate_ids)
        and footer.get("scored_count") == len(scores)
        and footer.get("valid_count") == valid_count
    )
    if output_scope == "all-candidates":
        scope_valid = (
            score_ids == candidate_id_set
            and len(scores) == len(candidate_ids)
            and evaluated_valid_count == valid_count
        )
    elif output_scope == "selected-top-k":
        local_top_k = footer.get("top_k_requested")
        shortlist = footer.get("shortlist")
        scope_valid = (
            score_ids <= candidate_id_set
            and valid_count == len(scores)
            and footer.get("evaluated_count") == len(candidate_ids)
            and isinstance(evaluated_valid_count, int)
            and not isinstance(evaluated_valid_count, bool)
            and 0 <= evaluated_valid_count <= len(candidate_ids)
            and isinstance(local_top_k, int)
            and not isinstance(local_top_k, bool)
            and local_top_k >= required_top_k
            and len(scores) == min(local_top_k, evaluated_valid_count)
            and isinstance(shortlist, list)
            and len(shortlist) == len(scores)
            and all(
                isinstance(item, dict)
                and item.get("rank") == rank
                and item.get("candidate_id") == scores[rank].get("candidate_id")
                and item.get("predicted_scheduler_makespan")
                == scores[rank].get("predicted_scheduler_makespan")
                for rank, item in enumerate(shortlist)
            )
        )
    else:
        scope_valid = False
    if not common_counts_valid or not scope_valid:
        raise RankingError("score footer counts are inconsistent")
    return header, scores, int(evaluated_valid_count)


def build_global_ranking(
    graph_manifest: Path,
    architecture: Path,
    variant_artifacts: Sequence[VariantArtifacts],
    communication_mode: str,
    top_k: int,
    max_total_candidates: int,
    placement_search_policy: str = "exact",
    activity_scheduling_policy: str = "fixed",
    activity_beam_width: int = 1,
    selected_graph_variant_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if communication_mode not in ("none", "explicit"):
        raise RankingError("communication mode is invalid")
    if placement_search_policy not in PLACEMENT_POLICY_RECORD:
        raise RankingError("placement search policy is invalid")
    if activity_scheduling_policy not in ACTIVITY_POLICY_RECORD:
        raise RankingError("activity scheduling policy is invalid")
    if activity_beam_width <= 0:
        raise RankingError("activity beam width must be positive")
    if top_k <= 0 or max_total_candidates <= 0:
        raise RankingError("ranking bounds must be positive")
    graph_header, variants = load_graph_variants(graph_manifest)
    if selected_graph_variant_ids is not None:
        selected_ids = list(selected_graph_variant_ids)
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise RankingError("selected graph identities must be non-empty and unique")
        available_ids = {variant.graph_variant_id for variant in variants}
        missing_ids = sorted(set(selected_ids) - available_ids)
        if missing_ids:
            raise RankingError(
                "selected graph identities are absent from the manifest: "
                + ", ".join(missing_ids)
            )
        selected_id_set = set(selected_ids)
        variants = [
            variant
            for variant in variants
            if variant.graph_variant_id in selected_id_set
        ]
        if [variant.graph_variant_id for variant in variants] != selected_ids:
            raise RankingError(
                "selected graph identities must follow graph manifest order"
            )
    if [item.graph for item in variant_artifacts] != variants:
        raise RankingError("variant artifacts must cover graph order exactly")
    architecture_sha = sha256_file(architecture)
    candidate_spaces: List[Tuple[VariantArtifacts, List[str]]] = []
    total_candidates = 0
    for item in variant_artifacts:
        _, candidate_ids = _load_candidates(item.candidates, item.graph)
        total_candidates += len(candidate_ids)
        candidate_spaces.append((item, candidate_ids))
    if total_candidates > max_total_candidates:
        raise RankingError(
            "total candidate budget exceeded before score validation"
        )

    ranked: List[Tuple[int, int, int, Dict[str, Any]]] = []
    artifact_rows = []
    valid_candidates = 0
    for item, candidate_ids in candidate_spaces:
        _, scores, evaluated_valid_count = _load_scores(
            item.scores,
            item.candidates,
            item.costs,
            item.graph,
            architecture_sha,
            communication_mode,
            top_k,
            placement_search_policy,
            activity_scheduling_policy,
            activity_beam_width,
        )
        valid_candidates += evaluated_valid_count
        candidate_manifest_sha256 = sha256_file(item.candidates)
        cost_catalog_sha256 = sha256_file(item.costs)
        score_manifest_sha256 = sha256_file(item.scores)
        artifact_rows.append(
            {
                "graph_variant_id": item.graph.graph_variant_id,
                "graph_variant_index": item.graph.index,
                "result_ir_sha256": item.graph.ir_sha256,
                "candidate_manifest_sha256": candidate_manifest_sha256,
                "cost_catalog_sha256": cost_catalog_sha256,
                "score_manifest_sha256": score_manifest_sha256,
            }
        )
        candidate_index = {value: index for index, value in enumerate(candidate_ids)}
        for score in scores:
            if score.get("valid") is not True:
                continue
            candidate_id = str(score["candidate_id"])
            makespan = score.get("predicted_scheduler_makespan")
            if (
                isinstance(makespan, bool)
                or not isinstance(makespan, (int, float))
                or not math.isfinite(float(makespan))
                or makespan <= 0
            ):
                raise RankingError("valid score has an invalid makespan")
            if float(makespan) != int(makespan):
                raise RankingError("scheduler makespan must be a whole cycle")
            trace_sha = _sha(
                score.get("schedule_trace_sha256"), "schedule trace SHA-256"
            )
            row = {
                "graph_variant_id": item.graph.graph_variant_id,
                "graph_variant_index": item.graph.index,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index[candidate_id],
                "predicted_scheduler_makespan": int(makespan),
                "schedule_trace_sha256": trace_sha,
                "candidate_manifest_sha256": candidate_manifest_sha256,
                "cost_catalog_sha256": cost_catalog_sha256,
                "score_manifest_sha256": score_manifest_sha256,
            }
            ranked.append(
                (
                    int(makespan),
                    item.graph.index,
                    candidate_index[candidate_id],
                    row,
                )
            )
    if not ranked:
        raise RankingError("no valid whole-program candidate")
    ranked.sort(key=lambda value: value[:3])
    shortlist = []
    for rank, (_, _, _, row) in enumerate(ranked[: min(top_k, len(ranked))]):
        shortlist.append({"rank": rank, **row})
    return {
        "schema": GLOBAL_SCHEMA,
        "complete": True,
        "graph_manifest_sha256": sha256_file(graph_manifest),
        "selected_graph_variant_ids": [
            variant.graph_variant_id for variant in variants
        ],
        "graph_source": graph_header["source"],
        "architecture_sha256": architecture_sha,
        "communication_mode": communication_mode,
        "placement_search_policy": placement_search_policy,
        "activity_scheduling_policy": activity_scheduling_policy,
        "activity_beam_width": activity_beam_width,
        "bounds": {
            "top_k": top_k,
            "max_total_candidates": max_total_candidates,
        },
        "counts": {
            "graph_variants": len(variants),
            "total_candidates": total_candidates,
            "valid_candidates": valid_candidates,
        },
        "variant_artifacts": artifact_rows,
        "shortlist": shortlist,
    }


def write_global_ranking(path: Path, ranking: Mapping[str, Any]) -> None:
    if ranking.get("schema") != GLOBAL_SCHEMA or ranking.get("complete") is not True:
        raise RankingError("refusing to publish an incomplete global ranking")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(ranking, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "GLOBAL_SCHEMA",
    "GraphVariant",
    "RankingError",
    "VariantArtifacts",
    "build_global_ranking",
    "load_graph_variants",
    "sha256_file",
    "write_global_ranking",
]
