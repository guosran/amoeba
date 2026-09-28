#!/usr/bin/env python3
"""Independent reference model for the canonical ORBIT joint rewrite pass.

This model checks the small producer/consumer pipeline used by the joint
tiling/fusion review::

    T = C_initial + A * B
    D = T + E

The representation is deliberately a data model rather than an MLIR printer.
The C++ MLIR passes own production rewrite actions, legality, graph identity,
and materialization.  This Python model is an independent regression oracle.
A graph is JSON serialisable.  Action history is retained only for diagnostics
and is excluded from the graph identity.

The public API is intentionally small:

``root_graph(identity)``
    Construct the source-bound, unmodified task graph.
``apply_action(graph, action)``
    Apply one typed action atomically and return a ``RewriteActionResult``.
``validate_graph(graph)``
    Return a boolean semantic validation result.  ``return_audit=True`` gives
    stable reason codes and diagnostics.
``canonical_graph_id(graph)``
    Hash the result graph, never the rewrite history or runtime inputs.
``enumerate_rewrite_dag(graph)``
    Explore accepted and rejected actions with memoised canonical states.
``interpret_graph(graph, values)``
    Evaluate T and D with i32 modulo-2**32 arithmetic and debug reductions.

No backend, hardware, mapping, cost, or scheduling information is accepted by
the graph identity.  The model is intentionally conservative: unsupported
accesses, aliasing, shape changes, side effects, and reduction consumers fail
closed with stable reason codes.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA = "orbit-joint-semantic-graph-v1"
ACTION_SCHEMA = "orbit-joint-rewrite-action-v1"
REGION_SCHEMA = "orbit-half-open-region-v1"
ARITHMETIC_SEMANTICS = "i32_modulo_2^32"

TASK_KINDS = {
    "gemm_producer",
    "elementwise_consumer",
    "fused_producer_consumer",
    "reduction",
    "fused_reduction_consumer",
    "completion_join",
}

ACTION_KINDS = (
    "TileOutputM",
    "TileOutputN",
    "TileReductionKSequential",
    "TileReductionKParallelLinear",
    "TileReductionKParallelTree",
    "FuseProducerConsumer",
    "FuseTileLocalPair",
    "FuseReductionConsumer",
    "MaterializeCompletionJoin",
    "MaterializeNumericReduction",
)

REJECTION_CODES = (
    "ACTION_UNKNOWN",
    "ACTION_NOOP",
    "ACTION_TARGET_UNKNOWN",
    "ALREADY_TILED",
    "TILE_FACTOR_INVALID",
    "TILE_TAIL_REJECTED",
    "K_POLICY_CONFLICT",
    "FUSION_ALREADY_PRESENT",
    "FUSION_REQUIRES_TILING",
    "REDUCTION_REQUIRED",
    "REDUCTION_NOT_MATERIALIZED",
    "NUMERIC_REDUCTION_ALREADY_PRESENT",
    "COMPLETION_JOIN_ALREADY_PRESENT",
    "EXTERNAL_USER_PRESENT",
    "ALIAS_PROOF_MISSING",
    "SIDE_EFFECT_UNPROVEN",
    "UNKNOWN_ACCESS",
    "MULTIPLE_EXITS",
    "DYNAMIC_CONTROL",
    "REGION_MISMATCH",
    "HALO_UNSUPPORTED",
    "BROADCAST_UNSUPPORTED",
    "RESHAPE_UNSUPPORTED",
    "REDUCTION_CONSUMER_UNSUPPORTED",
    "PARTIAL_CONSUMER_DIRECT",
    "PARTIAL_OWNERSHIP_INVALID",
    "STATE_CHAIN_INVALID",
    "INITIAL_C_DUPLICATED",
    "CONSUMER_DUPLICATED",
    "OUTPUT_OVERLAP",
    "OUTPUT_HOLE",
    "TASK_ID_INVALID",
    "EDGE_ENDPOINT_INVALID",
    "EDGE_REGION_INVALID",
    "TASK_KIND_INVALID",
    "INCOMPLETE_REDUCTION",
    "F32_REASSOCIATION_UNPROVEN",
    "GRAPH_INVALID",
)


class JointRewriteError(ValueError):
    """A named semantic graph or action failure."""

    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


class GraphValidationError(JointRewriteError):
    """Raised when a graph fails a semantic proof obligation."""


class RewriteActionResult(dict):
    """Mapping result returned by :func:`apply_action`.

    The result deliberately exposes both ``result["graph"]`` and
    ``result.graph``.  For ergonomic callers, graph keys are delegated when a
    transition was accepted, so ``validate_graph(result)`` is also supported.
    """

    @property
    def accepted(self) -> bool:
        return bool(dict.get(self, "accepted", False))

    @property
    def graph(self) -> Mapping[str, Any] | None:
        return dict.get(self, "graph")

    @property
    def reason_code(self) -> str | None:
        return dict.get(self, "reason_code")

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, key: str) -> Any:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            graph = dict.get(self, "graph")
            if self.accepted and isinstance(graph, Mapping) and key in graph:
                return graph[key]
            raise

    def get(self, key: str, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        graph = dict.get(self, "graph")
        if self.accepted and isinstance(graph, Mapping):
            return graph.get(key, default)
        return default


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fail(code: str, message: str) -> None:
    raise JointRewriteError(code, message)


def _unwrap_graph(graph: Any) -> Mapping[str, Any]:
    if isinstance(graph, RewriteActionResult):
        if not graph.accepted or not isinstance(graph.graph, Mapping):
            _fail("GRAPH_INVALID", "action result does not contain an accepted graph")
        return graph.graph
    if isinstance(graph, Mapping) and isinstance(graph.get("graph"), Mapping) \
            and "tasks" not in graph:
        return graph["graph"]
    if not isinstance(graph, Mapping):
        _fail("GRAPH_INVALID", "graph must be a mapping")
    return graph


def _as_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("TILE_FACTOR_INVALID", f"{name} must be an integer >= {minimum}")
    return value


def _region(buffer: str, lower: Sequence[int], upper: Sequence[int],
            access: str, **extra: Any) -> dict[str, Any]:
    if len(lower) != len(upper) or any(
            isinstance(lo, bool) or isinstance(hi, bool) or
            not isinstance(lo, int) or not isinstance(hi, int) or
            lo < 0 or hi < lo for lo, hi in zip(lower, upper)):
        _fail("EDGE_REGION_INVALID", "region bounds must be non-negative half-open integers")
    result: dict[str, Any] = {
        "schema": REGION_SCHEMA,
        "buffer": str(buffer),
        "lower": [int(v) for v in lower],
        "upper": [int(v) for v in upper],
        "half_open": True,
        "access": str(access),
    }
    result.update(extra)
    return result


def _edge(source: str, target: str, kind: str, scope: str,
          region: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_task": source,
        "target_task": target,
        "kind": kind,
        "type": kind,
        "scope": scope,
        "edge_scope": scope,
    }
    if region is not None:
        result["region"] = _json_copy(region)
    result.update(extra)
    return result


def _tiles(extent: int, factor: int) -> list[list[int]]:
    return [[start, min(start + factor, extent)]
            for start in range(0, extent, factor)]


def _full_region(graph: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = graph["dimensions"]
    return _region("T", [0, 0], [dimensions["M"], dimensions["N"]],
                   "read_write", tensor="T")


def _identity_source(program: Any) -> dict[str, Any]:
    """Normalize a pipeline identity without making MLIR parsing mandatory."""

    if program is None:
        identity = {
            "schema": "orbit-canonical-producer-consumer-pipeline-v1",
            "canonical_pipeline_id": "canonical-producer-consumer-8x8x8-i32-v1",
            "source_module_sha256": "canonical-producer-consumer-8x8x8-i32-v1",
            "function": "gemm_pipeline",
            "M": 8,
            "N": 8,
            "K": 8,
            "dtype": "i32",
            "layout": "row_major_identity",
            "arithmetic_semantics": ARITHMETIC_SEMANTICS,
            "alias_proof": "five distinct amoeba.noalias arguments plus internal T allocation",
        }
        identity["canonical_source_id"] = identity["canonical_pipeline_id"]
        return identity

    raw: Any = program
    if isinstance(raw, Mapping):
        for key in ("pipeline_identity", "canonical_pipeline_identity",
                    "program_identity", "descriptor"):
            if isinstance(raw.get(key), Mapping):
                raw = raw[key]
                break
        if not isinstance(raw, Mapping):
            _fail("GRAPH_INVALID", "pipeline identity must be a mapping")
        identity = dict(raw)
    elif isinstance(raw, (Path, str)):
        path = Path(raw)
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                _fail("SOURCE_READ_FAILED", str(error))
            identity = {
                "source_module_sha256": _sha256(text),
                "source_path": str(path),
            }
            # Reuse the strict recognizer when it is available beside this
            # module.  Keeping this optional makes the graph model useful in
            # unit tests that provide a compact identity mapping.
            try:
                recognizer = importlib.import_module("orbit_joint_pipeline")
            except (ImportError, AttributeError) as error:
                _fail("SOURCE_RECOGNIZER_UNAVAILABLE", str(error))
            # Validation errors from the strict recognizer intentionally
            # propagate.  A real source file must never silently become the
            # default 8x8 identity after an unknown access or alias proof.
            identity = dict(recognizer.canonical_pipeline_identity(path))
        else:
            identity = {"source_module_sha256": _sha256(str(raw))}
    else:
        _fail("GRAPH_INVALID", "program must be a source path or identity mapping")

    defaults = {
        "schema": "orbit-canonical-producer-consumer-pipeline-v1",
        "function": "gemm_pipeline",
        "M": 8,
        "N": 8,
        "K": 8,
        "dtype": "i32",
        "layout": "row_major_identity",
        "arithmetic_semantics": ARITHMETIC_SEMANTICS,
        "alias_proof": "explicit noalias",
    }
    for key, value in defaults.items():
        identity.setdefault(key, value)
    for key in ("M", "N", "K"):
        identity[key] = _as_int(identity[key], f"program.{key}", minimum=1)
    if identity.get("layout") == "row_major":
        identity["layout"] = "row_major_identity"
    if identity.get("layout") != "row_major_identity":
        _fail("RESHAPE_UNSUPPORTED", "only row-major identity layout is supported")
    if identity.get("dtype") != "i32":
        _fail("F32_REASSOCIATION_UNPROVEN",
              "joint semantic evaluator currently proves i32 modulo arithmetic only")
    if "arithmetic_semantics" not in identity:
        identity["arithmetic_semantics"] = ARITHMETIC_SEMANTICS
    source_id = (identity.get("canonical_pipeline_id") or
                 identity.get("canonical_program_id") or
                 identity.get("source_module_sha256") or
                 identity.get("source_program_sha256"))
    if not isinstance(source_id, str) or not source_id:
        source_id = _sha256({key: identity.get(key) for key in (
            "function", "M", "N", "K", "dtype", "layout",
            "arithmetic_semantics", "alias_proof")})
    identity["canonical_source_id"] = source_id
    return _json_copy(identity)


def _default_contracts() -> dict[str, Any]:
    return {
        "alias_free": True,
        "producer_side_effect_free": True,
        "consumer_side_effect_free": True,
        "producer_unknown_access": False,
        "consumer_unknown_access": False,
        "multiple_exits": False,
        "dynamic_control": False,
        "producer_consumer_region": "exact",
        "consumer_shape": "exact",
        "consumer_halo": False,
        "consumer_broadcast": False,
        "consumer_reshape": False,
        "consumer_is_reduction": False,
        "intermediate_external_users": [],
        "intermediate_aliases": [],
        "numeric_reassociation_proven": True,
        "taskflow_completion_only_join": True,
    }


def _task(task_id: str, task_kind: str, role: str,
          tile_id: str, m_range: Sequence[int], n_range: Sequence[int],
          k_range: Sequence[int] | None, regions: Mapping[str, Any],
          graph: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    current_fusion = graph.get("fusion", {})
    if (current_fusion.get("producer_consumer") in ("whole", "tile_local") or
            current_fusion.get("sequential_last") or
            current_fusion.get("reduction_consumer")):
        default_visibility = "fused_internal"
    else:
        default_visibility = "global_materialized"
    region_values = _json_copy(regions)
    read_buffers: list[str] = []
    write_buffers: list[str] = []
    for region in region_values.values():
        region_items = region if isinstance(region, list) else [region]
        for item in region_items:
            if not isinstance(item, Mapping):
                continue
            buffer = str(item.get("buffer", ""))
            access = str(item.get("access", ""))
            if (item.get("visibility") == "fused_internal" and
                    buffer == "fused_internal_T"):
                # T is eliminated into the fused body.  Keep its grouped
                # region for semantic identity, but do not advertise it as a
                # taskflow operand or external read.
                continue
            if access in ("read", "private_read", "read_modify_write"):
                if buffer not in read_buffers:
                    read_buffers.append(buffer)
            if access in ("write", "private_write", "read_modify_write"):
                if buffer not in write_buffers:
                    write_buffers.append(buffer)
    result: dict[str, Any] = {
        "task_id": task_id,
        "task_kind": task_kind,
        "task_role": role,
        "semantic_task": True,
        "logical_output_tile_id": tile_id,
        "m_range": list(m_range),
        "n_range": list(n_range),
        "k_range": list(k_range) if k_range is not None else None,
        "M_range": list(m_range),
        "N_range": list(n_range),
        "K_range": list(k_range) if k_range is not None else None,
        "regions": _json_copy(regions),
        "reads": read_buffers,
        "writes": write_buffers,
        "will_reads": list(read_buffers),
        "will_writes": list(write_buffers),
        "original_read_memrefs": list(read_buffers),
        "original_write_memrefs": list(write_buffers),
        "k_policy": graph["k_policy"],
        "reduction_topology": graph["reduction_topology"],
        # ``intermediate`` is derived after all tasks are rebuilt.  Reading a
        # stale value here would make equivalent rewrite orders produce
        # different task identities.
        "intermediate_visibility": default_visibility,
        "predecessors": [],
        "successors": [],
    }
    for region_name in ("A_read", "B_read", "C_input", "T_output",
                        "partial_output", "T_read", "E_read", "D_output",
                        "partial_inputs", "C_initial_input"):
        if region_name in region_values:
            result[region_name + "_region"] = _json_copy(region_values[region_name])
    result.update(extra)
    return result


def _tile_id(m_range: Sequence[int], n_range: Sequence[int]) -> str:
    return f"tile_m{m_range[0]}_n{n_range[0]}"


def _k_id(index: int, k_range: Sequence[int]) -> str:
    return f"k{index}_{k_range[0]}_{k_range[1]}"


def _output_tiles(graph: Mapping[str, Any]) -> list[tuple[str, list[int], list[int]]]:
    return [(_tile_id(m_range, n_range), list(m_range), list(n_range))
            for m_range in graph["m_tiles"] for n_range in graph["n_tiles"]]


def _regions_for_producer(graph: Mapping[str, Any], m_range: Sequence[int],
                          n_range: Sequence[int], k_range: Sequence[int],
                          *, first: bool, parallel: bool,
                          fused: bool, partial_name: str | None = None) -> dict[str, Any]:
    m0, m1 = m_range
    n0, n1 = n_range
    k0, k1 = k_range
    regions: dict[str, Any] = {
        "A_read": _region("A", [m0, k0], [m1, k1], "read"),
        "B_read": _region("B", [k0, n0], [k1, n1], "read"),
        "C_input": None,
        "T_output": None,
        "partial_output": None,
    }
    if parallel:
        regions["partial_output"] = _region(
            partial_name or "partial_T", [m0, n0], [m1, n1], "private_write",
            logical_output_tile_id=_tile_id(m_range, n_range),
            logical_k_block=_k_id(0, k_range))
    else:
        c_buffer = "C_initial" if first else "state_carry"
        regions["C_input"] = _region(
            c_buffer, [m0, n0], [m1, n1], "read",
            ownership="initial_C_once" if first else "previous_K_block")
        if fused:
            regions["T_output"] = _region(
                "D", [m0, n0], [m1, n1], "write", visibility="fused_internal")
        else:
            regions["T_output"] = _region(
                "T", [m0, n0], [m1, n1], "write")
    return regions


def _regions_for_consumer(graph: Mapping[str, Any], m_range: Sequence[int],
                          n_range: Sequence[int], *, fused: bool) -> dict[str, Any]:
    m0, m1 = m_range
    n0, n1 = n_range
    input_buffer = "fused_internal_T" if fused else "T"
    return {
        "T_read": _region(input_buffer, [m0, n0], [m1, n1], "read",
                           visibility="fused_internal" if fused else "global"),
        "E_read": _region("E", [m0, n0], [m1, n1], "read"),
        "D_output": _region("D", [m0, n0], [m1, n1], "write"),
    }


def _regions_for_reduction(graph: Mapping[str, Any], m_range: Sequence[int],
                           n_range: Sequence[int], k_blocks: Sequence[Sequence[int]],
                           *, fused: bool) -> dict[str, Any]:
    m0, m1 = m_range
    n0, n1 = n_range
    partials = [_region(
        f"partial_T/{_tile_id(m_range, n_range)}/{_k_id(index, block)}",
        [m0, n0], [m1, n1], "private_read",
        logical_output_tile_id=_tile_id(m_range, n_range),
        logical_k_block=_k_id(index, block))
                for index, block in enumerate(k_blocks)]
    return {
        "partial_inputs": partials,
        "C_initial_input": _region("C_initial", [m0, n0], [m1, n1], "read",
                                     ownership="initial_C_once"),
        "T_output": _region("D" if fused else "T", [m0, n0], [m1, n1],
                             "write", visibility="fused_internal" if fused else "global"),
    }


def _rebuild(graph: MutableMapping[str, Any]) -> None:
    """Derive tasks and typed edges from the semantic graph state."""

    m_tiles = graph["m_tiles"]
    n_tiles = graph["n_tiles"]
    k_blocks = graph["k_blocks"]
    policy = graph["k_policy"]
    topology = graph["reduction_topology"]
    fusion = graph["fusion"]
    output_tiles = _output_tiles(graph)
    tasks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    terminal_ids: list[str] = []
    parallel = policy == "parallel"

    producer_fused = fusion.get("producer_consumer") in ("whole", "tile_local")
    sequential_last_fused = fusion.get("sequential_last", False)
    reduction_fused = bool(fusion.get("reduction_consumer", False))

    # Fusion groups are result semantics, so derive them from the final tile
    # partition.  This normalization makes equivalent ``tile -> fuse`` and
    # ``fuse -> tile`` histories canonicalize to the same graph.
    if fusion.get("producer_consumer") == "tile_local":
        fusion["groups"] = [f"producer_consumer:{tile_id}"
                             for tile_id, _, _ in output_tiles]
    elif fusion.get("sequential_last"):
        fusion["groups"] = [f"producer_consumer:{tile_id}:last_k"
                             for tile_id, _, _ in output_tiles]
    elif fusion.get("reduction_consumer"):
        fusion["groups"] = [f"reduction_consumer:{tile_id}"
                             for tile_id, _, _ in output_tiles]
    elif fusion.get("producer_consumer") == "whole":
        fusion["groups"] = ["producer_consumer"]

    for tile_id, m_range, n_range in output_tiles:
        tile_producers: list[str] = []
        if policy == "parallel":
            for index, block in enumerate(k_blocks):
                kid = _k_id(index, block)
                task_id = f"partial_{tile_id}_{kid}"
                partial_name = f"partial_T/{tile_id}/{kid}"
                regions = _regions_for_producer(
                    graph, m_range, n_range, block, first=index == 0,
                    parallel=True, fused=False, partial_name=partial_name)
                task = _task(
                    task_id, "gemm_producer", "producer", tile_id,
                    m_range, n_range, block, regions, graph,
                    logical_k_block_id=kid,
                    output_visibility="private_partial",
                    partial_buffer=partial_name,
                    initial_C_ownership="reduction_task",
                    final_C_ownership="private_partial",
                    arithmetic_order=["m_ascending", "n_ascending", "k_ascending"],
                )
                tasks.append(task)
                tile_producers.append(task_id)

            if graph.get("numeric_reduction_materialized"):
                reduce_id = f"reduce_{tile_id}"
                if reduction_fused:
                    task = _task(
                        reduce_id, "fused_reduction_consumer",
                        "fused_reduction_consumer", tile_id, m_range, n_range,
                        [0, graph["dimensions"]["K"]],
                        _regions_for_reduction(graph, m_range, n_range,
                                               k_blocks, fused=True), graph,
                        numeric_reduction=True,
                        output_visibility="fused_internal",
                        reduction_topology=topology,
                        body=["reduce_private_partials", "add_E", "store_D"],
                    )
                    tasks.append(task)
                    terminal_ids.append(reduce_id)
                    target_id = reduce_id
                else:
                    task = _task(
                        reduce_id, "reduction", "reduction", tile_id, m_range,
                        n_range, [0, graph["dimensions"]["K"]],
                        _regions_for_reduction(graph, m_range, n_range,
                                               k_blocks, fused=False), graph,
                        numeric_reduction=True,
                        output_visibility="global_materialized",
                        reduction_topology=topology,
                        body=["reduce_private_partials", "store_T"],
                    )
                    tasks.append(task)
                    terminal_ids.append(reduce_id)
                    target_id = reduce_id
                for index, producer_id in enumerate(tile_producers):
                    block = k_blocks[index]
                    partial_name = f"partial_T/{tile_id}/{_k_id(index, block)}"
                    edges.append(_edge(
                        producer_id, target_id, "partial_reduction", "tile_local",
                        _region(partial_name, [m_range[0], n_range[0]],
                                [m_range[1], n_range[1]], "private_read",
                                logical_output_tile_id=tile_id,
                                logical_k_block=_k_id(index, block))))
                if not reduction_fused:
                    consumer_id = f"consumer_{tile_id}"
                    consumer = _task(
                        consumer_id, "elementwise_consumer", "consumer", tile_id,
                        m_range, n_range, None,
                        _regions_for_consumer(graph, m_range, n_range, fused=False), graph,
                        output_visibility="global_materialized",
                        body=["load_T", "load_E", "add", "store_D"],
                    )
                    tasks.append(consumer)
                    terminal_ids.append(consumer_id)
                    edges.append(_edge(
                        target_id, consumer_id, "producer_consumer",
                        "tile_local",
                        _region("T", [m_range[0], n_range[0]],
                                [m_range[1], n_range[1]], "read"), buffer="T"))
            continue

        # Sequential K (including the identity one-block case).
        for index, block in enumerate(k_blocks):
            kid = _k_id(index, block)
            is_last = index == len(k_blocks) - 1
            fused_here = bool(sequential_last_fused and is_last)
            if len(k_blocks) == 1 and producer_fused:
                fused_here = True
            if producer_fused and len(k_blocks) > 1 and not sequential_last_fused:
                # A producer/consumer fusion request followed by K tiling is
                # normalized to the only legal sequential fusion point.
                fused_here = is_last
            task_id = (f"fused_{tile_id}_{kid}" if fused_here else
                       f"producer_{tile_id}_{kid}")
            regions = _regions_for_producer(
                graph, m_range, n_range, block, first=index == 0,
                parallel=False, fused=fused_here)
            if fused_here:
                regions.update(_regions_for_consumer(
                    graph, m_range, n_range, fused=True))
                task_kind = "fused_producer_consumer"
                role = "fused_producer_consumer"
                body = ["load_A", "load_B", "mul", "load_C_or_state",
                        "add_product", "load_E", "add_E", "store_D"]
                output_visibility = "fused_internal"
            else:
                task_kind = "gemm_producer"
                role = "producer"
                body = ["load_A", "load_B", "mul", "load_C_or_state",
                        "add_product", "store_T"]
                output_visibility = "global_materialized"
            task = _task(
                task_id, task_kind, role, tile_id, m_range, n_range, block,
                regions, graph, logical_k_block_id=kid,
                output_visibility=output_visibility,
                initial_C_ownership="initial_C_once" if index == 0 else "state_carry",
                final_C_ownership=("external_final_D" if fused_here else
                                   "external_final_C" if is_last else "state_carry"),
                body=body,
            )
            tasks.append(task)
            tile_producers.append(task_id)
            if index:
                previous_id = tile_producers[index - 1]
                edges.append(_edge(
                    previous_id, task_id, "state_carry", "tile_local",
                    _region("state_carry", [m_range[0], n_range[0]],
                            [m_range[1], n_range[1]], "read_write",
                            ownership="previous_K_block"), buffer="state_carry"))
        last_id = tile_producers[-1]
        if not any(task["task_id"] == last_id and
                   task["task_kind"] == "fused_producer_consumer" for task in tasks):
            consumer_id = f"consumer_{tile_id}"
            consumer = _task(
                consumer_id, "elementwise_consumer", "consumer", tile_id,
                m_range, n_range, None,
                _regions_for_consumer(graph, m_range, n_range, fused=False), graph,
                output_visibility="global_materialized",
                body=["load_T", "load_E", "add", "store_D"],
            )
            tasks.append(consumer)
            terminal_ids.append(consumer_id)
            edges.append(_edge(
                last_id, consumer_id, "producer_consumer",
                "tile_local",
                _region("T", [m_range[0], n_range[0]],
                        [m_range[1], n_range[1]], "read"), buffer="T"))
        else:
            terminal_ids.append(last_id)

    if graph.get("completion_join_materialized"):
        join_id = "completion_join"
        join = _task(
            join_id, "completion_join", "completion_join", "all_outputs",
            [0, graph["dimensions"]["M"]], [0, graph["dimensions"]["N"]], None,
            {"D_output": _region("D", [0, 0],
                                   [graph["dimensions"]["M"], graph["dimensions"]["N"]],
                                   "completion", completion_only=True)}, graph,
            completion_only=True,
            numeric=False,
            body=["completion_only"],
            output_visibility="external_final_D",
        )
        tasks.append(join)
        for terminal in terminal_ids:
            edges.append(_edge(
                terminal, join_id, "completion", "tensor_wide",
                _region("D", [0, 0],
                        [graph["dimensions"]["M"], graph["dimensions"]["N"]],
                        "completion", completion_only=True)))

    task_ids = {task["task_id"] for task in tasks}
    predecessors: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    successors: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for edge in edges:
        source = edge["source_task"]
        target = edge["target_task"]
        if source in successors and target in predecessors:
            successors[source].add(target)
            predecessors[target].add(source)
    for task in tasks:
        task["predecessors"] = sorted(predecessors[task["task_id"]])
        task["successors"] = sorted(successors[task["task_id"]])
    graph["tasks"] = tasks
    graph["edges"] = edges
    graph["terminal_task_ids"] = terminal_ids
    graph["intermediate"] = _derive_intermediate(graph)
    graph["output_tiles"] = [
        {
            "logical_output_tile_id": tile_id,
            "m_range": list(m_range),
            "n_range": list(n_range),
            "D_region": _region("D", list(m_range[:1]) + list(n_range[:1]),
                                 list(m_range[1:]) + list(n_range[1:]), "write"),
        }
        for tile_id, m_range, n_range in output_tiles
    ]
    graph["k_block_records"] = [
        {"logical_k_block_id": _k_id(index, block), "index": index,
         "k_range": list(block)}
        for index, block in enumerate(k_blocks)
    ]
    graph["tile_sizes"] = {
        "BM": max((tile[1] - tile[0] for tile in m_tiles), default=0),
        "BN": max((tile[1] - tile[0] for tile in n_tiles), default=0),
        "BK": max((block[1] - block[0] for block in k_blocks), default=0),
    }
    graph["variant_id"] = canonical_graph_id(graph)
    graph["semantic_legal"] = not (
        graph["k_policy"] == "parallel" and
        not graph.get("numeric_reduction_materialized"))
    graph["completion_task_id"] = (
        "completion_join" if graph.get("completion_join_materialized") else None)
    graph["fusion_mode"] = (
        "reduction_consumer" if fusion.get("reduction_consumer") else
        "sequential_last" if fusion.get("sequential_last") else
        fusion.get("producer_consumer", "none"))
    graph["graph_id"] = canonical_graph_id(graph)
    graph["mode_tags"] = _mode_tags(graph)
    graph["classification"] = graph["mode_tags"]["primary"]


def _derive_intermediate(graph: Mapping[str, Any]) -> dict[str, Any]:
    fusion = graph["fusion"]
    parallel = graph["k_policy"] == "parallel"
    fused = bool(fusion.get("producer_consumer") in ("whole", "tile_local") or
                 fusion.get("sequential_last") or
                 fusion.get("reduction_consumer"))
    if fused:
        visibility = "fused_internal"
        allocation = "eliminated"
        lifetime = "within_fused_task"
    else:
        visibility = "global_materialized"
        allocation = "global_T"
        lifetime = "producer_to_consumer"
    partial_buffers: list[str] = []
    if parallel:
        for tile_id, m_range, n_range in _output_tiles(graph):
            for index, block in enumerate(graph["k_blocks"]):
                partial_buffers.append(
                    f"partial_T/{tile_id}/{_k_id(index, block)}")
    return {
        "identity": "T",
        "visibility": visibility,
        "state": "eliminated" if allocation == "eliminated" else "global_materialized",
        "allocation": allocation,
        "lifetime": lifetime,
        "external_users": list(graph["contracts"].get("intermediate_external_users", [])),
        "partial_buffers": partial_buffers,
        "private_partial": bool(partial_buffers),
        "producer": "producer",
        "consumer": "consumer",
    }


def _mode_tags(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a result graph without changing its canonical identity."""

    tiled_m = len(graph.get("m_tiles", [])) > 1
    tiled_n = len(graph.get("n_tiles", [])) > 1
    tiled_k = len(graph.get("k_blocks", [])) > 1
    tiled = tiled_m or tiled_n or tiled_k
    fusion = graph.get("fusion", {})
    fused = bool(fusion.get("producer_consumer") in ("whole", "tile_local") or
                 fusion.get("sequential_last") or
                 fusion.get("reduction_consumer"))
    tags: list[str] = []
    if not tiled and not fused:
        tags.append("identity")
    if tiled and not fused:
        tags.append("tiling_only")
    if fused and not tiled:
        tags.append("fusion_only")
    history = graph.get("action_history", [])
    history_kinds = [item.get("kind") for item in history
                     if isinstance(item, Mapping)]
    tile_positions = [index for index, item in enumerate(history_kinds)
                      if isinstance(item, str) and item.startswith("Tile")]
    fusion_positions = [index for index, item in enumerate(history_kinds)
                        if item in ("FuseProducerConsumer", "FuseTileLocalPair",
                                    "FuseReductionConsumer")]
    if tile_positions and fusion_positions:
        if min(tile_positions) < max(fusion_positions):
            tags.append("tile_then_fuse")
        if min(fusion_positions) < max(tile_positions):
            tags.append("fuse_then_tile")
    if tiled_k and graph.get("k_policy") == "sequential" and \
            fusion.get("sequential_last"):
        tags.append("sequential_last_fusion")
    if graph.get("k_policy") == "parallel" and graph.get("numeric_reduction_materialized") and \
            fusion.get("reduction_consumer"):
        if graph.get("reduction_topology") == "linear":
            tags.append("parallel_linear_reduction_fusion")
        elif graph.get("reduction_topology") == "balanced_tree":
            tags.append("parallel_tree_reduction_fusion")
    if tiled_m and tiled_n and tiled_k and fused:
        tags.append("mnk_fusion")
    if not tags:
        tags.append("semantic_rewrite")
    # Preserve a deterministic order while keeping the most specific mode
    # useful to callers as ``primary``.
    priority = ["identity", "fusion_only", "tiling_only", "tile_then_fuse",
                "fuse_then_tile", "sequential_last_fusion",
                "parallel_linear_reduction_fusion",
                "parallel_tree_reduction_fusion", "mnk_fusion"]
    primary = next((item for item in priority if item in tags), tags[0])
    return {"primary": primary, "modes": sorted(set(tags)), "tiled": tiled,
            "fused": fused}


def classify_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic semantic mode tags for a graph state."""

    return _mode_tags(_unwrap_graph(graph))


def root_graph(program: Any = None, *, contract_overrides: Mapping[str, Any] | None = None,
               m_tile: int | None = None, n_tile: int | None = None) -> dict[str, Any]:
    """Create the source-bound canonical graph with no rewrites applied."""

    identity = _identity_source(program)
    dimensions = {key: int(identity[key]) for key in ("M", "N", "K")}
    graph: dict[str, Any] = {
        "schema": SCHEMA,
        "graph_kind": "semantic_task_graph",
        "source_identity": identity,
        "canonical_source_id": identity["canonical_source_id"],
        "canonical_pipeline_id": identity.get("canonical_pipeline_id",
                                               identity["canonical_source_id"]),
        "canonical_program_id": identity.get("canonical_program_id",
                                              identity["canonical_source_id"]),
        "source_module_sha256": identity.get("source_module_sha256"),
        "source_program_sha256": identity.get("source_program_sha256",
                                               identity.get("source_module_sha256")),
        "program_identity": identity,
        "dimensions": dimensions,
        "M": dimensions["M"],
        "N": dimensions["N"],
        "K": dimensions["K"],
        "dtype": identity.get("dtype", "i32"),
        "layout": identity.get("layout", "row_major_identity"),
        "arithmetic_semantics": identity.get("arithmetic_semantics", ARITHMETIC_SEMANTICS),
        "tail_policy": "allow",
        "m_tiles": [[0, dimensions["M"]]],
        "n_tiles": [[0, dimensions["N"]]],
        "k_blocks": [[0, dimensions["K"]]],
        "k_policy": "sequential",
        "reduction_topology": "none",
        "numeric_reduction_materialized": False,
        "completion_join_materialized": False,
        "fusion": {
            "producer_consumer": "none",
            "sequential_last": False,
            "reduction_consumer": False,
            "groups": [],
        },
        "contracts": _default_contracts(),
        "action_history": [],
        "source_provenance": {
            "canonical_source_id": identity["canonical_source_id"],
            "program_schema": identity.get("schema"),
            "source_module_sha256": identity.get("source_module_sha256"),
        },
    }
    if contract_overrides:
        graph["contracts"].update(_json_copy(contract_overrides))
    if m_tile is not None:
        graph["m_tiles"] = _tiles(dimensions["M"], _as_int(m_tile, "m_tile", 1))
    if n_tile is not None:
        graph["n_tiles"] = _tiles(dimensions["N"], _as_int(n_tile, "n_tile", 1))
    _rebuild(graph)
    return graph


def _semantic_graph_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Select only result semantics for canonical graph identity."""

    graph = _unwrap_graph(graph)
    task_payload: list[dict[str, Any]] = []
    for task in sorted(graph.get("tasks", []), key=lambda value: value.get("task_id", "")):
        task_payload.append({
            key: _json_copy(task[key]) for key in (
                "task_id", "task_kind", "task_role", "logical_output_tile_id",
                "logical_k_block_id", "m_range", "n_range", "k_range", "regions",
                "k_policy", "reduction_topology", "intermediate_visibility",
                "output_visibility", "initial_C_ownership", "final_C_ownership",
                "numeric_reduction", "completion_only", "body")
            if key in task
        })
    edge_payload = [_json_copy(edge) for edge in sorted(
        graph.get("edges", []), key=lambda value: _canonical(value))]
    identity = graph.get("program_identity", graph.get("source_identity", {}))
    identity_payload = {
        key: _json_copy(identity[key]) for key in (
            "canonical_source_id", "canonical_pipeline_id", "canonical_program_id",
            "source_module_sha256", "source_program_sha256", "function", "M", "N",
            "K", "dtype", "layout", "arithmetic_semantics", "alias_proof")
        if key in identity
    }
    return {
        "schema": SCHEMA,
        "canonical_source_id": graph.get("canonical_source_id"),
        "program_identity": identity_payload,
        "dimensions": _json_copy(graph.get("dimensions")),
        "dtype": graph.get("dtype"),
        "layout": graph.get("layout"),
        "arithmetic_semantics": graph.get("arithmetic_semantics"),
        "m_tiles": _json_copy(graph.get("m_tiles")),
        "n_tiles": _json_copy(graph.get("n_tiles")),
        "k_blocks": _json_copy(graph.get("k_blocks")),
        "k_policy": graph.get("k_policy"),
        "reduction_topology": graph.get("reduction_topology"),
        "numeric_reduction_materialized": bool(graph.get("numeric_reduction_materialized")),
        "completion_join_materialized": bool(graph.get("completion_join_materialized")),
        "fusion": _json_copy(graph.get("fusion")),
        "intermediate": _json_copy(graph.get("intermediate")),
        "tasks": task_payload,
        "edges": edge_payload,
    }


def canonical_graph_id(graph: Mapping[str, Any]) -> str:
    """Return an action-history and runtime-independent semantic graph ID."""

    return "graph-" + _sha256(_semantic_graph_payload(graph))


def _audit(valid: bool, reason_code: str | None = None,
           diagnostic: str | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": bool(valid),
        "reason_code": reason_code,
        "diagnostic": diagnostic,
    }
    result.update(extra)
    return result


def _check_region(region: Mapping[str, Any], graph: Mapping[str, Any], *,
                  allowed_buffers: Iterable[str] | None = None) -> None:
    if region.get("half_open") is not True or region.get("schema") != REGION_SCHEMA:
        _fail("EDGE_REGION_INVALID", "region must use explicit half-open schema")
    lower = region.get("lower")
    upper = region.get("upper")
    if not isinstance(lower, list) or not isinstance(upper, list) or len(lower) != len(upper):
        _fail("EDGE_REGION_INVALID", "region bounds are malformed")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in lower + upper):
        _fail("EDGE_REGION_INVALID", "region bounds must be non-negative integers")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        _fail("EDGE_REGION_INVALID", "region lower bound exceeds upper bound")
    dimensions = graph["dimensions"]
    buffer_name = str(region.get("buffer", ""))
    if buffer_name == "A":
        limits = [dimensions["M"], dimensions["K"]]
    elif buffer_name == "B":
        limits = [dimensions["K"], dimensions["N"]]
    else:
        limits = [dimensions["M"], dimensions["N"]]
    if len(upper) == 2 and any(hi > limit for hi, limit in zip(upper, limits)):
        _fail("EDGE_REGION_INVALID", "region exceeds output domain")
    if allowed_buffers is not None and region.get("buffer") not in set(allowed_buffers):
        # Private partial names and fused aliases are valid as long as their
        # prefix is explicit and cannot masquerade as a global tensor.
        buffer = str(region.get("buffer", ""))
        if not (buffer.startswith("partial_T/") or buffer in {"fused_internal_T", "state_carry"}):
            _fail("EDGE_REGION_INVALID", f"unknown region buffer {buffer!r}")


def _contract_failure(graph: Mapping[str, Any]) -> tuple[str, str] | None:
    contracts = graph.get("contracts", {})
    checks = (
        (not contracts.get("alias_free", True), "ALIAS_PROOF_MISSING",
         "alias freedom is not proven"),
        (bool(contracts.get("intermediate_aliases")), "ALIAS_PROOF_MISSING",
         "intermediate T aliases another buffer"),
        (not contracts.get("producer_side_effect_free", True), "SIDE_EFFECT_UNPROVEN",
         "producer has an unproven side effect"),
        (not contracts.get("consumer_side_effect_free", True), "SIDE_EFFECT_UNPROVEN",
         "consumer has an unproven side effect"),
        (contracts.get("producer_unknown_access", False), "UNKNOWN_ACCESS",
         "producer has an unknown access"),
        (contracts.get("consumer_unknown_access", False), "UNKNOWN_ACCESS",
         "consumer has an unknown access"),
        (contracts.get("multiple_exits", False), "MULTIPLE_EXITS",
         "task has multiple exits"),
        (contracts.get("dynamic_control", False), "DYNAMIC_CONTROL",
         "task control is dynamic"),
        (contracts.get("producer_consumer_region") not in (None, "exact", "compatible"),
         "REGION_MISMATCH", "producer and consumer regions do not match"),
        (contracts.get("consumer_shape") not in (None, "exact", "compatible"),
         "REGION_MISMATCH", "consumer shape is incompatible"),
        (contracts.get("consumer_halo", False), "HALO_UNSUPPORTED",
         "halo access cannot be tile-localized"),
        (contracts.get("consumer_broadcast", False), "BROADCAST_UNSUPPORTED",
         "broadcast access is not exact tile-local access"),
        (contracts.get("consumer_reshape", False), "RESHAPE_UNSUPPORTED",
         "reshape changes the typed region"),
        (contracts.get("consumer_is_reduction", False), "REDUCTION_CONSUMER_UNSUPPORTED",
         "reduction consumer requires a separate proof"),
        (bool(contracts.get("intermediate_external_users")), "EXTERNAL_USER_PRESENT",
         "intermediate T has an unaccounted-for external user"),
    )
    for predicate, code, message in checks:
        if predicate:
            return code, message
    return None


def _validate_graph_impl(graph: Mapping[str, Any], *, allow_incomplete: bool) -> None:
    required = ("schema", "dimensions", "m_tiles", "n_tiles", "k_blocks",
                "k_policy", "reduction_topology", "fusion", "tasks", "edges",
                "intermediate", "contracts")
    missing = [key for key in required if key not in graph]
    if missing:
        _fail("GRAPH_INVALID", f"graph misses fields {missing!r}")
    if graph.get("schema") != SCHEMA:
        _fail("GRAPH_INVALID", "graph schema is unknown")
    dimensions = graph["dimensions"]
    for key in ("M", "N", "K"):
        if dimensions.get(key) != graph.get(key, dimensions.get(key)):
            _fail("GRAPH_INVALID", f"dimension {key} is inconsistent")
        _as_int(dimensions.get(key), f"dimensions.{key}", 1)
    failure = _contract_failure(graph)
    if failure:
        _fail(*failure)
    if graph.get("k_policy") not in ("sequential", "parallel"):
        _fail("K_POLICY_CONFLICT", "unknown K policy")
    if graph.get("k_policy") == "sequential" and graph.get("reduction_topology") != "none":
        _fail("K_POLICY_CONFLICT", "sequential K cannot have a parallel topology")
    if graph.get("k_policy") == "parallel" and graph.get("reduction_topology") not in (
            "linear", "balanced_tree"):
        _fail("K_POLICY_CONFLICT", "parallel K requires linear or balanced_tree topology")
    if graph.get("k_policy") == "sequential" and graph.get("reduction_topology") != "none":
        _fail("K_POLICY_CONFLICT", "sequential K requires topology none")
    m_tiles = graph.get("m_tiles")
    n_tiles = graph.get("n_tiles")
    k_blocks = graph.get("k_blocks")
    for name, values, extent in (("m_tiles", m_tiles, dimensions["M"]),
                                 ("n_tiles", n_tiles, dimensions["N"]),
                                 ("k_blocks", k_blocks, dimensions["K"])):
        if not isinstance(values, list) or not values:
            _fail("OUTPUT_HOLE", f"{name} must be a non-empty partition")
        previous = 0
        for interval in values:
            if not isinstance(interval, list) or len(interval) != 2:
                _fail("EDGE_REGION_INVALID", f"{name} interval is malformed")
            lo, hi = interval
            if lo != previous or not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
                _fail("OUTPUT_HOLE", f"{name} does not form a contiguous partition")
            if hi > extent:
                _fail("EDGE_REGION_INVALID", f"{name} exceeds domain")
            previous = hi
        if previous != extent:
            _fail("OUTPUT_HOLE", f"{name} does not cover the complete domain")
    tasks = graph.get("tasks")
    edges = graph.get("edges")
    if not isinstance(tasks, list) or not isinstance(edges, list):
        _fail("GRAPH_INVALID", "tasks and edges must be lists")
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, Mapping)]
    if len(task_ids) != len(set(task_ids)) or any(not isinstance(value, str) or not value for value in task_ids):
        _fail("TASK_ID_INVALID", "task identifiers must be unique non-empty strings")
    task_map = {task["task_id"]: task for task in tasks}
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("task_kind") not in TASK_KINDS:
            _fail("TASK_KIND_INVALID", "task kind is not in the semantic vocabulary")
        for region in task.get("regions", {}).values():
            if isinstance(region, list):
                for item in region:
                    if isinstance(item, Mapping):
                        _check_region(item, graph)
            elif isinstance(region, Mapping):
                _check_region(region, graph)
        if task.get("task_kind") == "completion_join":
            if task.get("completion_only") is not True or task.get("numeric") is not False:
                _fail("GRAPH_INVALID", "completion join must be completion-only")
    for edge in edges:
        if not isinstance(edge, Mapping):
            _fail("EDGE_ENDPOINT_INVALID", "edge is not a mapping")
        source, target = edge.get("source_task"), edge.get("target_task")
        if source not in task_map or target not in task_map:
            _fail("EDGE_ENDPOINT_INVALID", "edge endpoint is not a task")
        if not edge.get("kind") or edge.get("type") != edge.get("kind"):
            _fail("EDGE_ENDPOINT_INVALID", "edge kind/type must be explicit and equal")
        if edge.get("scope") not in ("tile_local", "tensor_wide", "private_partial"):
            _fail("EDGE_ENDPOINT_INVALID", "edge scope is unknown")
        if isinstance(edge.get("region"), Mapping):
            _check_region(edge["region"], graph)
    # A parallel graph is an intermediate construction state only until the
    # numeric reduction action has materialized its reduction task.
    if graph.get("k_policy") == "parallel" and not graph.get("numeric_reduction_materialized"):
        if not allow_incomplete:
            _fail("INCOMPLETE_REDUCTION", "parallel partials have no numeric reduction")
    if graph.get("numeric_reduction_materialized"):
        expected = "fused_reduction_consumer" if graph["fusion"].get("reduction_consumer") \
            else "reduction"
        reduction_tasks = [task for task in tasks if task.get("task_kind") in (
            "reduction", "fused_reduction_consumer")]
        if len(reduction_tasks) != len(_output_tiles(graph)):
            _fail("INCOMPLETE_REDUCTION", "one reduction task per output tile is required")
        if any(task.get("task_kind") != expected for task in reduction_tasks):
            _fail("INCOMPLETE_REDUCTION", "reduction fusion state and task kind disagree")
        for edge in edges:
            if edge.get("kind") == "partial_reduction":
                if task_map[edge["source_task"]].get("task_kind") != "gemm_producer":
                    _fail("PARTIAL_OWNERSHIP_INVALID", "partial edge source is not a producer")
                if task_map[edge["target_task"]].get("task_kind") not in (
                        "reduction", "fused_reduction_consumer"):
                    _fail("PARTIAL_OWNERSHIP_INVALID", "partial edge target is not a reduction")
    for edge in edges:
        if edge.get("kind") == "producer_consumer":
            source = task_map[edge["source_task"]]
            target = task_map[edge["target_task"]]
            if source.get("task_kind") in ("gemm_producer",) and \
                    source.get("k_policy") == "parallel":
                _fail("PARTIAL_CONSUMER_DIRECT", "partial K producer cannot feed consumer directly")
            if target.get("task_kind") != "elementwise_consumer":
                _fail("REGION_MISMATCH", "producer-consumer edge target is not a consumer")
    if graph["k_policy"] == "sequential":
        _validate_state_chain(graph)
    # Every semantic graph returned by a rewrite is complete except the
    # explicit parallel-construction state used between the two typed actions.
    if graph.get("completion_join_materialized") and not graph.get("terminal_task_ids"):
        _fail("GRAPH_INVALID", "completion join has no terminal task")


def _validate_state_chain(graph: Mapping[str, Any]) -> None:
    tasks = graph["tasks"]
    by_tile: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        if task.get("task_kind") in ("gemm_producer", "fused_producer_consumer"):
            by_tile.setdefault(task["logical_output_tile_id"], []).append(task)
    for tile_id, tile_tasks in by_tile.items():
        tile_tasks = sorted(tile_tasks, key=lambda task: task.get("k_range") or [0, 0])
        if len(tile_tasks) != len(graph["k_blocks"]):
            _fail("STATE_CHAIN_INVALID", f"tile {tile_id} has incomplete K chain")
        for index, task in enumerate(tile_tasks):
            expected = "initial_C_once" if index == 0 else "state_carry"
            if task.get("initial_C_ownership") != expected:
                _fail("INITIAL_C_DUPLICATED" if index else "STATE_CHAIN_INVALID",
                      f"tile {tile_id} has incorrect C ownership")
            if index == len(tile_tasks) - 1 and task.get("task_kind") == "fused_producer_consumer":
                continue
            if index == len(tile_tasks) - 1 and task.get("final_C_ownership") != "external_final_C":
                _fail("STATE_CHAIN_INVALID", f"tile {tile_id} does not publish final T")
        consumer_count = sum(1 for task in tasks
                             if task.get("task_kind") == "elementwise_consumer" and
                             task.get("logical_output_tile_id") == tile_id)
        if consumer_count > 1:
            _fail("CONSUMER_DUPLICATED", f"tile {tile_id} has multiple consumers")
        if any(task.get("task_kind") == "fused_producer_consumer"
               for task in tile_tasks[:-1]):
            _fail("CONSUMER_DUPLICATED", f"consumer fused before final K block for {tile_id}")


def validate_graph(graph: Mapping[str, Any], *, raise_on_error: bool = False,
                   return_audit: bool = False, allow_incomplete: bool = False) -> bool | dict[str, Any]:
    """Validate a semantic graph and return ``True``/``False``.

    ``return_audit`` gives a stable record with ``reason_code``.  The
    ``allow_incomplete`` option is for the explicit intermediate state after
    parallel K construction and before numeric reduction materialization.
    """

    try:
        graph_map = _unwrap_graph(graph)
        _validate_graph_impl(graph_map, allow_incomplete=allow_incomplete)
    except JointRewriteError as error:
        if raise_on_error:
            raise GraphValidationError(error.reason_code, str(error)) from error
        result = _audit(False, error.reason_code, str(error))
        return result if return_audit else False
    result = _audit(True, None, None,
                    graph_id=canonical_graph_id(graph_map),
                    task_count=len(graph_map.get("tasks", [])),
                    edge_count=len(graph_map.get("edges", [])))
    return result if return_audit else True


def _action_parts(action: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(action, str):
        return action, {"kind": action}
    if isinstance(action, Mapping):
        kind = action.get("kind", action.get("action_kind", action.get("action")))
        if not isinstance(kind, str):
            _fail("ACTION_UNKNOWN", "action kind is missing")
        parameters = dict(action)
        if isinstance(action.get("parameters"), Mapping):
            parameters.update(action["parameters"])
        parameters["kind"] = kind
        return kind, parameters
    if isinstance(action, (tuple, list)) and action and isinstance(action[0], str):
        parameters = dict(action[1]) if len(action) > 1 and isinstance(action[1], Mapping) else {}
        parameters["kind"] = action[0]
        return action[0], parameters
    _fail("ACTION_UNKNOWN", "action must be a kind string or mapping")


def _factor(parameters: Mapping[str, Any], extent: int, axis: str,
            *, allow_exact: bool = False) -> int:
    raw = (parameters.get("tile_size") if parameters.get("tile_size") is not None else
           parameters.get("factor", parameters.get("size", parameters.get(axis))))
    if raw is None:
        raw = max(1, extent // 2)
    factor = _as_int(raw, f"{axis} tile factor", 1)
    if factor > extent:
        _fail("TILE_FACTOR_INVALID", f"{axis} tile factor exceeds the domain")
    if factor == extent and not allow_exact:
        _fail("ACTION_NOOP", f"{axis} tile factor does not split the domain")
    return factor


def _check_action_contracts(graph: Mapping[str, Any]) -> None:
    failure = _contract_failure(graph)
    if failure:
        _fail(*failure)
    if graph["contracts"].get("intermediate_external_users"):
        _fail("EXTERNAL_USER_PRESENT", "T has an external user")
    if graph["contracts"].get("numeric_reassociation_proven") is False and \
            graph.get("k_policy") == "parallel":
        _fail("F32_REASSOCIATION_UNPROVEN",
              "parallel reduction reassociation is not proven")


def _target_check(graph: Mapping[str, Any], parameters: Mapping[str, Any]) -> None:
    target_ids = parameters.get("target_task_ids", parameters.get("target_tasks"))
    if target_ids is None:
        return
    if isinstance(target_ids, str):
        target_ids = [target_ids]
    if not isinstance(target_ids, Sequence):
        _fail("ACTION_TARGET_UNKNOWN", "target_task_ids must be a sequence")
    known = {task.get("task_id") for task in graph.get("tasks", [])}
    if any(target not in known for target in target_ids):
        _fail("ACTION_TARGET_UNKNOWN", "action target task is not in the graph")


def _apply_to_state(graph: MutableMapping[str, Any], kind: str,
                    parameters: Mapping[str, Any]) -> None:
    dimensions = graph["dimensions"]
    fusion = graph["fusion"]
    if kind == "TileOutputM":
        _check_action_contracts(graph)
        if len(graph["m_tiles"]) != 1:
            _fail("ALREADY_TILED", "M output tiling is already present")
        factor = _factor(parameters, dimensions["M"], "M")
        graph["m_tiles"] = _tiles(dimensions["M"], factor)
        if fusion.get("producer_consumer") == "whole":
            fusion["producer_consumer"] = "tile_local"
        return
    if kind == "TileOutputN":
        _check_action_contracts(graph)
        if len(graph["n_tiles"]) != 1:
            _fail("ALREADY_TILED", "N output tiling is already present")
        factor = _factor(parameters, dimensions["N"], "N")
        graph["n_tiles"] = _tiles(dimensions["N"], factor)
        if fusion.get("producer_consumer") == "whole":
            fusion["producer_consumer"] = "tile_local"
        return
    if kind in ("TileReductionKSequential", "TileReductionKParallelLinear",
                "TileReductionKParallelTree"):
        _check_action_contracts(graph)
        if len(graph["k_blocks"]) != 1:
            _fail("ALREADY_TILED", "K reduction tiling is already present")
        factor = _factor(
            parameters, dimensions["K"], "K",
            allow_exact=kind in ("TileReductionKParallelLinear",
                                 "TileReductionKParallelTree"))
        graph["k_blocks"] = _tiles(dimensions["K"], factor)
        if kind == "TileReductionKSequential":
            graph["k_policy"] = "sequential"
            graph["reduction_topology"] = "none"
            graph["numeric_reduction_materialized"] = False
            if fusion.get("producer_consumer") == "whole":
                fusion["producer_consumer"] = "none"
                fusion["sequential_last"] = True
            elif fusion.get("producer_consumer") == "tile_local":
                fusion["producer_consumer"] = "none"
                fusion["sequential_last"] = True
        else:
            graph["k_policy"] = "parallel"
            graph["reduction_topology"] = (
                "linear" if kind == "TileReductionKParallelLinear" else "balanced_tree")
            graph["numeric_reduction_materialized"] = False
            if fusion.get("producer_consumer") in ("whole", "tile_local"):
                # The relation is retained as an intent until a numeric
                # reduction exists.  The reduction consumer is the only task
                # that may then absorb E and produce D.
                fusion["producer_consumer"] = "none"
                fusion["consumer_intent"] = "reduction"
        return
    if kind == "FuseProducerConsumer":
        _check_action_contracts(graph)
        target_ids = parameters.get("target_task_ids", parameters.get("target_tasks"))
        if isinstance(target_ids, str):
            target_ids = [target_ids]
        if target_ids:
            task_map = {task["task_id"]: task for task in graph["tasks"]}
            for target_id in target_ids:
                target = task_map[target_id]
                k_range = target.get("k_range")
                if target.get("task_kind") == "gemm_producer" and \
                        isinstance(k_range, list) and k_range[1] != dimensions["K"]:
                    _fail("REGION_MISMATCH",
                          "a sequential K consumer can fuse only with the final K block")
        if fusion.get("producer_consumer") != "none" or fusion.get("sequential_last") or \
                fusion.get("reduction_consumer"):
            _fail("FUSION_ALREADY_PRESENT", "producer and consumer are already fused")
        if graph["k_policy"] == "parallel":
            _fail("REDUCTION_REQUIRED", "parallel K must reduce before consumer fusion")
        if len(graph["k_blocks"]) > 1:
            fusion["sequential_last"] = True
            fusion["producer_consumer"] = "none"
        elif len(graph["m_tiles"]) > 1 or len(graph["n_tiles"]) > 1:
            fusion["producer_consumer"] = "tile_local"
        else:
            fusion["producer_consumer"] = "whole"
        fusion["groups"] = ["producer_consumer"]
        return
    if kind == "FuseTileLocalPair":
        _check_action_contracts(graph)
        if len(graph["m_tiles"]) == 1 and len(graph["n_tiles"]) == 1:
            _fail("FUSION_REQUIRES_TILING", "tile-local fusion requires M or N tiling")
        if graph["k_policy"] == "parallel":
            _fail("REDUCTION_REQUIRED", "parallel partials require reduction-consumer fusion")
        if fusion.get("producer_consumer") != "none" or fusion.get("sequential_last"):
            _fail("FUSION_ALREADY_PRESENT", "tile-local fusion is already present")
        fusion["producer_consumer"] = "tile_local"
        fusion["groups"] = [
            f"producer_consumer:{tile_id}" for tile_id, _, _ in _output_tiles(graph)]
        if len(graph["k_blocks"]) > 1:
            fusion["producer_consumer"] = "none"
            fusion["sequential_last"] = True
        return
    if kind == "MaterializeNumericReduction":
        _check_action_contracts(graph)
        if graph["k_policy"] != "parallel":
            _fail("K_POLICY_CONFLICT", "numeric reduction requires parallel K")
        if graph.get("numeric_reduction_materialized"):
            _fail("NUMERIC_REDUCTION_ALREADY_PRESENT", "numeric reduction already exists")
        graph["numeric_reduction_materialized"] = True
        if fusion.get("consumer_intent") == "reduction":
            fusion.pop("consumer_intent", None)
        return
    if kind == "FuseReductionConsumer":
        _check_action_contracts(graph)
        if graph["k_policy"] == "sequential":
            if len(graph["k_blocks"]) == 1:
                if fusion.get("producer_consumer") != "none":
                    _fail("FUSION_ALREADY_PRESENT", "consumer is already fused")
                fusion["producer_consumer"] = (
                    "tile_local" if len(graph["m_tiles"]) > 1 or len(graph["n_tiles"]) > 1
                    else "whole")
                fusion["groups"] = ["producer_consumer"]
                return
            if fusion.get("sequential_last"):
                _fail("FUSION_ALREADY_PRESENT", "last K consumer is already fused")
            fusion["sequential_last"] = True
            fusion["groups"] = ["producer_consumer:last_k"]
            return
        if not graph.get("numeric_reduction_materialized"):
            _fail("REDUCTION_NOT_MATERIALIZED", "numeric reduction must be materialized first")
        if fusion.get("reduction_consumer"):
            _fail("FUSION_ALREADY_PRESENT", "reduction consumer is already fused")
        fusion["reduction_consumer"] = True
        fusion["groups"] = ["reduction_consumer"]
        return
    if kind == "MaterializeCompletionJoin":
        _check_action_contracts(graph)
        if graph.get("completion_join_materialized"):
            _fail("COMPLETION_JOIN_ALREADY_PRESENT", "completion join already exists")
        if graph["k_policy"] == "parallel" and not graph.get("numeric_reduction_materialized"):
            _fail("INCOMPLETE_REDUCTION", "completion requires a numeric reduction")
        graph["completion_join_materialized"] = True
        return
    _fail("ACTION_UNKNOWN", f"unsupported action {kind!r}")


def apply_action(graph: Mapping[str, Any], action: Any, *,
                 raise_on_reject: bool = False) -> RewriteActionResult:
    """Apply one typed action atomically.

    The input graph is never mutated.  Rejections include a stable
    ``reason_code`` and return no partially rewritten graph.
    """

    original = _unwrap_graph(graph)
    source_id = canonical_graph_id(original)
    try:
        kind, parameters = _action_parts(action)
        if kind not in ACTION_KINDS:
            _fail("ACTION_UNKNOWN", f"unsupported action {kind!r}")
        _target_check(original, parameters)
        candidate = copy.deepcopy(dict(original))
        candidate.setdefault("action_history", [])
        _apply_to_state(candidate, kind, parameters)
        candidate["action_history"].append({
            "kind": kind,
            "parameters": _json_copy(parameters),
            "source_graph_id": source_id,
        })
        _rebuild(candidate)
        # A parallel construction state is intentionally accepted here; all
        # published enumeration states use the complete validation path.
        _validate_graph_impl(candidate, allow_incomplete=(
            kind.startswith("TileReductionKParallel")))
        candidate["graph_id"] = canonical_graph_id(candidate)
        return RewriteActionResult({
            "schema": ACTION_SCHEMA,
            "accepted": True,
            "action": _json_copy(parameters),
            "source_graph_id": source_id,
            "result_graph_id": candidate["graph_id"],
            "prerequisites": _prerequisites(candidate, kind),
            "conflicts": _conflicts(candidate, kind),
            "affected_regions": _affected_regions(candidate, kind),
            "created_tasks": _task_delta(original, candidate, added=True),
            "removed_tasks": _task_delta(original, candidate, added=False),
            "created_edges": _edge_delta(original, candidate, added=True),
            "removed_edges": _edge_delta(original, candidate, added=False),
            "intermediate_lifetime": candidate["intermediate"],
            "graph": candidate,
            "reason_code": None,
            "diagnostic": None,
        })
    except JointRewriteError as error:
        if raise_on_reject:
            raise
        return RewriteActionResult({
            "schema": ACTION_SCHEMA,
            "accepted": False,
            "action": action,
            "source_graph_id": source_id,
            "result_graph_id": None,
            "prerequisites": [],
            "conflicts": [error.reason_code],
            "affected_regions": [],
            "created_tasks": [],
            "removed_tasks": [],
            "created_edges": [],
            "removed_edges": [],
            "intermediate_lifetime": original.get("intermediate"),
            "graph": None,
            "reason_code": error.reason_code,
            "diagnostic": str(error),
        })


def _prerequisites(graph: Mapping[str, Any], kind: str) -> list[str]:
    prerequisites: list[str] = []
    if kind in ("FuseTileLocalPair",):
        prerequisites.append("output_tiling_compatible")
    if kind == "FuseReductionConsumer":
        prerequisites.extend(["parallel_k", "numeric_reduction_materialized"])
    if kind == "MaterializeNumericReduction":
        prerequisites.extend(["parallel_k", "private_partial_buffers"])
    if kind == "MaterializeCompletionJoin":
        prerequisites.append("all_numeric_outputs_complete")
    if kind in ("FuseProducerConsumer", "FuseTileLocalPair", "FuseReductionConsumer"):
        prerequisites.append("exact_region_and_single_user")
    return prerequisites


def _conflicts(graph: Mapping[str, Any], kind: str) -> list[str]:
    conflicts: list[str] = []
    if kind.startswith("TileReductionKParallel"):
        conflicts.append("direct_final_C_writes")
    if kind in ("FuseProducerConsumer", "FuseTileLocalPair"):
        conflicts.extend(["surviving_external_T_user", "incompatible_region"])
    if kind == "FuseReductionConsumer":
        conflicts.append("partial_consumer_direct_fusion")
    return conflicts


def _affected_regions(graph: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    dimensions = graph["dimensions"]
    if kind == "TileOutputM":
        return [_region("T", [0, 0], [dimensions["M"], dimensions["N"]], "partition")]
    if kind == "TileOutputN":
        return [_region("T", [0, 0], [dimensions["M"], dimensions["N"]], "partition")]
    if kind.startswith("TileReductionK"):
        return [_region("A", [0, 0], [dimensions["M"], dimensions["K"]], "partition"),
                _region("B", [0, 0], [dimensions["K"], dimensions["N"]], "partition")]
    if kind == "MaterializeNumericReduction":
        return [_region("partial_T", [0, 0], [dimensions["M"], dimensions["N"]], "reduce")]
    return [_region("D", [0, 0], [dimensions["M"], dimensions["N"]], "read_write")]


def _task_delta(before: Mapping[str, Any], after: Mapping[str, Any], *, added: bool) -> list[str]:
    left = {task["task_id"] for task in before.get("tasks", [])}
    right = {task["task_id"] for task in after.get("tasks", [])}
    return sorted((right - left) if added else (left - right))


def _edge_delta(before: Mapping[str, Any], after: Mapping[str, Any], *, added: bool) -> list[dict[str, Any]]:
    left = {_canonical(edge): edge for edge in before.get("edges", [])}
    right = {_canonical(edge): edge for edge in after.get("edges", [])}
    values = right.keys() - left.keys() if added else left.keys() - right.keys()
    return [_json_copy((right if added else left)[key]) for key in sorted(values)]


def _action_catalog(graph: Mapping[str, Any], *, tile_m: int | None,
                    tile_n: int | None, tile_k: int | None,
                    m_tile_sizes: Iterable[int] | None = None,
                    n_tile_sizes: Iterable[int] | None = None,
                    k_tile_sizes: Iterable[int] | None = None) -> list[dict[str, Any]]:
    dimensions = graph["dimensions"]
    m_values = ([tile_m] if tile_m is not None else
                list(m_tile_sizes if m_tile_sizes is not None else (1, 2, 4, 8)))
    n_values = ([tile_n] if tile_n is not None else
                list(n_tile_sizes if n_tile_sizes is not None else (1, 2, 4, 8)))
    k_values = ([tile_k] if tile_k is not None else
                list(k_tile_sizes if k_tile_sizes is not None else (1, 2, 4, 8)))
    # Keep the factor set explicit in the manifest, while avoiding duplicate
    # action records when a caller supplies repeated values.
    m_values = sorted({int(value) for value in m_values})
    n_values = sorted({int(value) for value in n_values})
    k_values = sorted({int(value) for value in k_values})
    catalog: list[dict[str, Any]] = [{"kind": "FuseProducerConsumer"}]
    catalog.extend({"kind": "TileOutputM", "tile_size": value}
                   for value in m_values)
    catalog.extend({"kind": "TileOutputN", "tile_size": value}
                   for value in n_values)
    catalog.extend({"kind": "TileReductionKSequential", "tile_size": value}
                   for value in k_values)
    catalog.extend({"kind": "TileReductionKParallelLinear", "tile_size": value}
                   for value in k_values)
    catalog.extend({"kind": "TileReductionKParallelTree", "tile_size": value}
                   for value in k_values)
    catalog.extend([
        {"kind": "FuseTileLocalPair"},
        {"kind": "MaterializeNumericReduction"},
        {"kind": "FuseReductionConsumer"},
        {"kind": "MaterializeCompletionJoin"},
    ])
    return catalog


def enumerate_rewrite_dag(graph: Mapping[str, Any] | None = None, *,
                          max_depth: int = 8, max_actions: int = 2000,
                          tile_m: int | None = None, tile_n: int | None = None,
                          tile_k: int | None = None,
                          m_tile_sizes: Iterable[int] | None = None,
                          n_tile_sizes: Iterable[int] | None = None,
                          k_tile_sizes: Iterable[int] | None = None,
                          actions: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Enumerate the semantic rewrite DAG with canonical-state memoisation."""

    if max_depth < 0 or max_actions <= 0:
        _fail("GRAPH_INVALID", "enumeration budgets must be positive")
    root = root_graph() if graph is None else _unwrap_graph(graph)
    # Root graphs are expected to be complete.  A caller may pass a copied
    # graph with a stale derived task list; rebuilding keeps expansion stable.
    root = copy.deepcopy(dict(root))
    _rebuild(root)
    root_id = canonical_graph_id(root)
    action_catalog = [dict(action) for action in (actions or _action_catalog(
        root, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_tile_sizes=m_tile_sizes, n_tile_sizes=n_tile_sizes,
        k_tile_sizes=k_tile_sizes))]
    action_catalog.sort(key=lambda action: (_action_parts(action)[0], _canonical(action)))
    started = time.monotonic()
    queue: list[tuple[int, str, dict[str, Any], list[dict[str, Any]]]] = [
        (0, root_id, root, [])]
    seen: dict[str, dict[str, Any]] = {root_id: root}
    depth_by_id: dict[str, int] = {root_id: 0}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicate_count = 0
    attempted = 0
    max_frontier = 1
    while queue and attempted < max_actions:
        depth, current_id, current, history = queue.pop(0)
        if depth >= max_depth:
            continue
        for action in action_catalog:
            if attempted >= max_actions:
                break
            attempted += 1
            result = apply_action(current, action)
            record = {
                "action": _json_copy(action),
                "source_graph_id": current_id,
                "depth": depth + 1,
            }
            if result.accepted:
                next_graph = result.graph
                assert isinstance(next_graph, Mapping)
                next_id = result.result_graph_id
                record.update({
                    "accepted": True,
                    "result_graph_id": next_id,
                    "reason_code": None,
                })
                accepted.append(record)
                if next_id in seen:
                    duplicate_count += 1
                    record["duplicate_state"] = True
                    continue
                sequence = history + [_json_copy(action)]
                next_graph_copy = copy.deepcopy(dict(next_graph))
                seen[next_id] = next_graph_copy
                depth_by_id[next_id] = depth + 1
                queue.append((depth + 1, next_id, next_graph_copy, sequence))
                max_frontier = max(max_frontier, len(queue))
            else:
                record.update({
                    "accepted": False,
                    "result_graph_id": None,
                    "reason_code": result.reason_code,
                    "diagnostic": result.diagnostic,
                })
                rejected.append(record)
    elapsed = time.monotonic() - started
    state_list = [seen[graph_id] for graph_id in sorted(
        seen, key=lambda graph_id: (depth_by_id[graph_id], graph_id))]
    candidate_list = [item for item in state_list
                      if validate_graph(item)]
    mode_records = {
        graph_id: classify_graph(item) for graph_id, item in seen.items()
    }
    candidate_mode_records = {
        canonical_graph_id(item): classify_graph(item) for item in candidate_list
    }
    stats = {
        "raw_action_sequences": attempted,
        "action_attempts": attempted,
        "accepted_transitions": len(accepted),
        "rejected_transitions": len(rejected),
        "unique_graph_states": len(seen),
        "construction_states": len(state_list),
        "unique_candidate_graphs": len(candidate_list),
        "incomplete_states_removed": len(state_list) - len(candidate_list),
        "duplicate_states_removed": duplicate_count,
        "max_frontier": max_frontier,
        "max_depth": max_depth,
        "max_actions": max_actions,
        "enumeration_time_seconds": elapsed,
        "budget_exhausted": attempted >= max_actions,
    }
    return {
        "schema": "orbit-joint-rewrite-dag-v1",
        "root_graph_id": root_id,
        "graphs": candidate_list,
        "candidate_graphs": candidate_list,
        "states": state_list,
        "graph_ids": [canonical_graph_id(item) for item in candidate_list],
        "candidate_graph_ids": [canonical_graph_id(item) for item in candidate_list],
        "state_ids": [canonical_graph_id(item) for item in state_list],
        "modes": mode_records,
        "classifications": mode_records,
        "candidate_modes": candidate_mode_records,
        "accepted_transitions": accepted,
        "rejected_transitions": rejected,
        "statistics": stats,
        "stats": stats,
        "action_catalog": action_catalog,
    }


def _runtime_vector(values: Mapping[str, Any], name: str, size: int) -> list[int]:
    raw = values.get(name)
    if raw is None and name == "C_initial":
        raw = values.get("C")
    if raw is None:
        _fail("GRAPH_INVALID", f"runtime value {name} is missing")
    if isinstance(raw, (str, bytes)):
        _fail("GRAPH_INVALID", f"runtime value {name} is not a vector")
    try:
        vector = list(raw)
    except TypeError as error:
        raise JointRewriteError("GRAPH_INVALID", f"runtime value {name} is not iterable") from error
    if len(vector) != size or any(isinstance(item, bool) or not isinstance(item, int)
                                   or item < -(1 << 31) or item >= (1 << 32)
                                   for item in vector):
        _fail("GRAPH_INVALID", f"runtime value {name} has invalid i32 shape or word")
    return [item & 0xffffffff for item in vector]


def _add(left: int, right: int) -> int:
    return (left + right) & 0xffffffff


def _mul(left: int, right: int) -> int:
    return (left * right) & 0xffffffff


def _tile_values(vector: Sequence[int], n: int, m_range: Sequence[int],
                 n_range: Sequence[int]) -> list[int]:
    return [vector[row * n + column]
            for row in range(m_range[0], m_range[1])
            for column in range(n_range[0], n_range[1])]


def _write_tile(vector: MutableMapping[int, int] | list[int], n: int,
                m_range: Sequence[int], n_range: Sequence[int], values: Sequence[int]) -> None:
    index = 0
    for row in range(m_range[0], m_range[1]):
        for column in range(n_range[0], n_range[1]):
            vector[row * n + column] = values[index] & 0xffffffff
            index += 1


def _partial(a: Sequence[int], b: Sequence[int], m: int, n: int, k: int,
             m_range: Sequence[int], n_range: Sequence[int],
             k_range: Sequence[int]) -> list[int]:
    output: list[int] = []
    for row in range(m_range[0], m_range[1]):
        for column in range(n_range[0], n_range[1]):
            value = 0
            for kk in range(k_range[0], k_range[1]):
                value = _add(value, _mul(a[row * k + kk], b[kk * n + column]))
            output.append(value)
    return output


def _reduce_linear(partials: Sequence[Sequence[int]]) -> list[int]:
    if not partials:
        return []
    result = list(partials[0])
    for partial in partials[1:]:
        result = [_add(left, right) for left, right in zip(result, partial)]
    return result


def _reduce_tree(partials: Sequence[Sequence[int]]) -> tuple[list[int], list[list[list[int]]]]:
    if not partials:
        return [], []
    current = [list(item) for item in partials]
    levels: list[list[list[int]]] = [copy.deepcopy(current)]
    while len(current) > 1:
        next_level: list[list[int]] = []
        index = 0
        while index < len(current):
            if index + 1 == len(current):
                next_level.append(list(current[index]))
            else:
                next_level.append([_add(left, right)
                                   for left, right in zip(current[index], current[index + 1])])
            index += 2
        current = next_level
        levels.append(copy.deepcopy(current))
    return list(current[0]), levels


def interpret_graph(graph: Mapping[str, Any], values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Interpret a complete graph and expose intermediate numeric evidence."""

    graph = _unwrap_graph(graph)
    _validate_graph_impl(graph, allow_incomplete=False)
    dimensions = graph["dimensions"]
    m, n, k = dimensions["M"], dimensions["N"], dimensions["K"]
    runtime = values or {
        "A": [0] * (m * k), "B": [0] * (k * n),
        "C_initial": [0] * (m * n), "E": [0] * (m * n),
    }
    a = _runtime_vector(runtime, "A", m * k)
    b = _runtime_vector(runtime, "B", k * n)
    c_initial = _runtime_vector(runtime, "C_initial", m * n)
    e = _runtime_vector(runtime, "E", m * n)
    t = list(c_initial)
    d = [0] * (m * n)
    state_debug: dict[str, list[list[int]]] = {}
    partial_debug: dict[str, dict[str, list[int]]] = {}
    reduction_debug: dict[str, list[int]] = {}
    reduction_levels: dict[str, list[list[list[int]]]] = {}

    for tile_id, m_range, n_range in _output_tiles(graph):
        if graph["k_policy"] == "sequential":
            state = _tile_values(c_initial, n, m_range, n_range)
            states: list[list[int]] = [list(state)]
            for block in graph["k_blocks"]:
                part = _partial(a, b, m, n, k, m_range, n_range, block)
                state = [_add(left, right) for left, right in zip(state, part)]
                states.append(list(state))
            state_debug[tile_id] = states
            _write_tile(t, n, m_range, n_range, state)
        else:
            partials: list[list[int]] = []
            records: dict[str, list[int]] = {}
            for index, block in enumerate(graph["k_blocks"]):
                part = _partial(a, b, m, n, k, m_range, n_range, block)
                records[_k_id(index, block)] = list(part)
                partials.append(part)
            partial_debug[tile_id] = records
            if graph["reduction_topology"] == "linear":
                reduced = _reduce_linear(partials)
            else:
                reduced, levels = _reduce_tree(partials)
                reduction_levels[tile_id] = levels
            c_tile = _tile_values(c_initial, n, m_range, n_range)
            state = [_add(left, right) for left, right in zip(c_tile, reduced)]
            reduction_debug[tile_id] = list(state)
            _write_tile(t, n, m_range, n_range, state)
        e_tile = _tile_values(e, n, m_range, n_range)
        t_tile = _tile_values(t, n, m_range, n_range)
        d_tile = [_add(left, right) for left, right in zip(t_tile, e_tile)]
        _write_tile(d, n, m_range, n_range, d_tile)

    debug = {
        "state_carry": state_debug,
        "partials": partial_debug,
        "reduction_outputs": reduction_debug,
        "reduction_levels": reduction_levels,
        "intermediate": _json_copy(graph["intermediate"]),
        "k_policy": graph["k_policy"],
        "reduction_topology": graph["reduction_topology"],
    }
    return {
        "T": t,
        "D": d,
        "debug": debug,
        "partials": partial_debug,
        "state_carry": state_debug,
        "reduction_outputs": reduction_debug,
        "graph_id": canonical_graph_id(graph),
    }


__all__ = [
    "ACTION_KINDS",
    "ACTION_SCHEMA",
    "ARITHMETIC_SEMANTICS",
    "GraphValidationError",
    "JointRewriteError",
    "REGION_SCHEMA",
    "REJECTION_CODES",
    "RewriteActionResult",
    "SCHEMA",
    "apply_action",
    "classify_graph",
    "canonical_graph_id",
    "enumerate_rewrite_dag",
    "interpret_graph",
    "root_graph",
    "validate_graph",
]
