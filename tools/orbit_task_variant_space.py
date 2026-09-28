#!/usr/bin/env python3
"""Hardware independent GEMM task-variant enumeration and legalization.

This module is the semantic owner for the small GEMM task graph used by the
ORBIT generic-blocking work.  It intentionally knows nothing about a CGRA,
kernel artifacts, memory schedules, or costs.  A variant describes the
half-open regions and dependencies needed to compute one GEMM.  A later
backend pass may attach capability and artifact information to that record.

The representation is ordinary JSON-compatible dictionaries.  Keeping the
representation free of Python objects makes it suitable for manifest files
and also makes the identity calculation explicit and reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "orbit-semantic-gemm-task-variant-v1"
PROGRAM_SCHEMA = "orbit-canonical-gemm-program-v1"
REGION_SCHEMA = "orbit-half-open-region-v1"
ARITHMETIC_SEMANTICS = "i32_modulo_2^32"
SUPPORTED_K_POLICIES = ("sequential_accumulate", "parallel_partial_reduce")
SUPPORTED_TAIL_POLICIES = ("reject", "exact")
SUPPORTED_REDUCTION_TOPOLOGIES = ("none", "linear", "balanced_tree")
PARALLEL_REDUCTION_TOPOLOGIES = ("linear", "balanced_tree")


class VariantError(ValueError):
    """Raised when a program, variant, or runtime value is not legal."""


class VariantValidationError(VariantError):
    """Raised when a task variant fails independent semantic validation."""


def _canonical(value: Any) -> str:
    """Return the byte-stable JSON form used by all identities."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        value = _canonical(value).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VariantError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VariantError(f"{label} must be a non-negative integer")
    return value


def _normalise_factor_set(values: Iterable[int], label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise VariantError(f"{label} must be an iterable of positive integers")
    try:
        result = {_positive_int(value, label) for value in values}
    except TypeError as error:
        raise VariantError(f"{label} must be an iterable of positive integers") from error
    if not result:
        raise VariantError(f"{label} must not be empty")
    return tuple(sorted(result))


def _normalise_policy_set(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    try:
        result = set(values)
    except TypeError as error:
        raise VariantError("k_policies must be an iterable") from error
    unknown = result.difference(SUPPORTED_K_POLICIES)
    if unknown:
        raise VariantError(f"unsupported K policy: {sorted(unknown)!r}")
    if not result:
        raise VariantError("k_policies must not be empty")
    # Keep the user-facing order stable even if callers supplied a set.
    return tuple(policy for policy in SUPPORTED_K_POLICIES if policy in result)


def _normalise_tail_policy(value: str) -> str:
    # ``allow`` was used by an early prototype.  Accepting it at the API
    # boundary keeps callers source-compatible while identity uses one name.
    if value == "allow":
        value = "exact"
    if value not in SUPPORTED_TAIL_POLICIES:
        raise VariantError(
            f"tail_policy must be one of {SUPPORTED_TAIL_POLICIES!r}")
    return value


def _normalise_topology_set(values: Iterable[str] | str | None,
                            policies: Sequence[str]) -> tuple[str, ...]:
    """Normalize topology requests while preserving deterministic order.

    Sequential accumulation has exactly one semantic topology (``none``).
    When no topology is supplied, parallel reduction deliberately expands to
    both supported structures so the topology is part of the semantic search
    space rather than an implicit implementation detail.
    """

    if values is None:
        requested = set(SUPPORTED_REDUCTION_TOPOLOGIES)
    elif isinstance(values, str):
        requested = {values}
    else:
        try:
            requested = set(values)
        except TypeError as error:
            raise VariantError("reduction_topologies must be an iterable") from error
    unknown = requested.difference(SUPPORTED_REDUCTION_TOPOLOGIES)
    if unknown:
        raise VariantError(f"unsupported reduction topology: {sorted(unknown)!r}")
    if not requested:
        raise VariantError("reduction_topologies must not be empty")
    if "sequential_accumulate" in policies and "none" not in requested:
        raise VariantError("sequential_accumulate requires reduction topology none")
    if "parallel_partial_reduce" in policies and not requested.intersection(
            PARALLEL_REDUCTION_TOPOLOGIES):
        raise VariantError(
            "parallel_partial_reduce requires linear or balanced_tree topology")
    return tuple(topology for topology in SUPPORTED_REDUCTION_TOPOLOGIES
                 if topology in requested)


def _as_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return path
    return None


def _program_operation(dtype: str, alias_proof: str) -> dict[str, Any]:
    arithmetic = (ARITHMETIC_SEMANTICS if dtype == "i32"
                  else f"{dtype}_C_plus_A_times_B")
    return {
        "access": {
            "A": ["M", "K"],
            "B": ["K", "N"],
            "C_read": ["M", "N"],
            "C_write": ["M", "N"],
        },
        "arithmetic": arithmetic,
        "loop_order": ["M", "N", "K"],
        "alias": alias_proof,
    }


def _parse_canonical_mlir(path: Path) -> dict[str, Any]:
    """Parse the deliberately narrow canonical static GEMM source.

    This is a small source recognizer rather than a general MLIR parser.  It
    accepts the same canonical form as the existing ORBIT recognizer and
    rejects unknown effects or dynamic shapes before a semantic variant can be
    made.
    """

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise VariantError(f"cannot read GEMM source {path}: {error}") from error
    if "linalg.matmul" in source or "affine.for" in source:
        raise VariantError("only canonical static SCF GEMM source is accepted")
    signature = re.search(
        r"func\.func\s+@([A-Za-z_][\w.]*)\s*\((.*?)\)\s*\{",
        source, flags=re.S)
    if signature is None or source.count("func.func") != 1:
        raise VariantError("source must contain exactly one GEMM function")
    function = signature.group(1)
    signature_args = [part.strip() for part in signature.group(2).split(",")]
    if len(signature_args) != 3:
        raise VariantError("source must have exactly three GEMM operands")
    args = re.findall(
        r"(%\w+)\s*:\s*memref<(\d+)x(\d+)x([A-Za-z_][\w]*)>\s*"
        r"\{amoeba\.noalias\}", signature.group(2))
    if len(args) != 3 or len(re.findall(r"memref<", signature.group(2))) != 3:
        raise VariantError("source must have three static noalias memrefs")
    (a_name, am, ak, a_dtype), (b_name, bk, bn, b_dtype), (c_name, cm, cn,
                                                             c_dtype) = args
    if len({a_name, b_name, c_name}) != 3:
        raise VariantError("operand alias is not proven")
    if not (a_dtype == b_dtype == c_dtype):
        raise VariantError("GEMM operands have different element types")
    m, n, k = int(am), int(bn), int(ak)
    if (int(bk), int(cm), int(cn)) != (k, m, n):
        raise VariantError("source shapes are not A[M,K], B[K,N], C[M,N]")
    # Parse the operation stream line by line.  A search for the expected
    # accesses is insufficient here: an unknown side effect or a second store
    # must not be hidden between two recognized operations.
    lines = [line.strip() for line in source.splitlines()
             if line.strip() and not line.strip().startswith("//")]
    if len(lines) < 10 or lines[0] != "module {" or lines[-1] != "}":
        raise VariantError("canonical module braces are required")
    if not re.fullmatch(
            r"func\.func\s+@" + re.escape(function) + r"\s*\(.*\)\s*\{",
            lines[1]):
        raise VariantError("canonical function signature is malformed")
    if lines[-2] != "}":
        raise VariantError("canonical function terminator is missing")
    ops = list(lines[2:-2])
    constants: dict[str, int] = {}
    defined_values = {a_name, b_name, c_name}
    while ops:
        constant = re.fullmatch(r"(%\w+) = arith\.constant (\d+) : index",
                                ops[0])
        if constant is None:
            break
        name, value = constant.groups()
        if name in defined_values:
            raise VariantError("duplicate index constant")
        constants[name] = int(value)
        defined_values.add(name)
        ops.pop(0)
    if not {0, 1, m, n, k}.issubset(constants.values()):
        raise VariantError("static loop bound constants are missing")
    loop_vars: list[str] = []
    for bound in (m, n, k):
        if not ops:
            raise VariantError("source must have three nested loops")
        match = re.fullmatch(
            r"scf\.for (%\w+) = (%\w+) to (%\w+) step (%\w+) \{",
            ops.pop(0))
        if match is None:
            raise VariantError("source must have three nested unit-step loops")
        iv, lower, upper, step = match.groups()
        if (constants.get(lower) != 0 or constants.get(upper) != bound or
                constants.get(step) != 1 or iv in defined_values):
            raise VariantError("SCF loop bounds or order are not proven")
        loop_vars.append(iv)
        defined_values.add(iv)
    mi, ni, ki = loop_vars
    load_patterns = [
        rf"(%\w+) = memref\.load {re.escape(a_name)}\[{re.escape(mi)}, {re.escape(ki)}\] : memref<{m}x{k}x{re.escape(a_dtype)}>",
        rf"(%\w+) = memref\.load {re.escape(b_name)}\[{re.escape(ki)}, {re.escape(ni)}\] : memref<{k}x{n}x{re.escape(a_dtype)}>",
        rf"(%\w+) = memref\.load {re.escape(c_name)}\[{re.escape(mi)}, {re.escape(ni)}\] : memref<{m}x{n}x{re.escape(a_dtype)}>",
    ]
    loaded: list[str] = []
    for pattern in load_patterns:
        if not ops:
            raise VariantError("GEMM load is missing")
        match = re.fullmatch(pattern, ops.pop(0))
        if match is None:
            raise VariantError("GEMM load access relation is not proven")
        value = match.group(1)
        if value in defined_values:
            raise VariantError("GEMM load must define a fresh SSA value")
        loaded.append(value)
        defined_values.add(value)
    aa, bb, cc = loaded
    if len(set(loaded)) != 3:
        raise VariantError("GEMM loads must define distinct SSA values")
    multiply_op, add_op = (("muli", "addi") if a_dtype == "i32"
                            else ("mulf", "addf"))
    if not ops:
        raise VariantError("GEMM multiply is missing")
    product = re.fullmatch(
        rf"(%\w+) = arith\.{multiply_op} {re.escape(aa)}, {re.escape(bb)} : {re.escape(a_dtype)}",
        ops.pop(0))
    if product is None:
        raise VariantError("GEMM multiply operation is not proven")
    if product.group(1) in defined_values:
        raise VariantError("GEMM multiply must define a fresh SSA value")
    defined_values.add(product.group(1))
    if not ops:
        raise VariantError("GEMM add is missing")
    updated = re.fullmatch(
        rf"(%\w+) = arith\.{add_op} {re.escape(cc)}, {re.escape(product.group(1))} : {re.escape(a_dtype)}",
        ops.pop(0))
    if updated is None:
        raise VariantError("C read-modify-write add is not proven")
    if updated.group(1) in defined_values:
        raise VariantError("C add must define a fresh SSA value")
    defined_values.add(updated.group(1))
    expected_store = (
        rf"memref\.store {re.escape(updated.group(1))}, {re.escape(c_name)}"
        rf"\[{re.escape(mi)}, {re.escape(ni)}\] : memref<{m}x{n}x{re.escape(a_dtype)}>"
    )
    if not ops or re.fullmatch(expected_store, ops.pop(0)) is None:
        raise VariantError("C store access relation is not proven")
    # Three loop closes and return are the complete remainder.  In particular,
    # this rejects a second store, unknown call, or side effect.
    if ops != ["}", "}", "}", "return"]:
        raise VariantError("unknown operation or side effect in GEMM body")
    alias_proof = "three distinct amoeba.noalias args"
    # Keep the source identity byte-compatible with the existing ORBIT
    # recognizer.  The textual relation is still checked above; this compact
    # operation record is what the current sidecar binds into its identity.
    operation = {
        "access": {
            "A": ["M", "K"],
            "B": ["K", "N"],
            "C_read": ["M", "N"],
            "C_write": ["M", "N"],
        },
        "arithmetic": "i32.wrap(C+A*B)" if a_dtype == "i32"
        else f"{a_dtype}.C_plus_A_times_B",
        "loop_order": ["M", "N", "K"],
        "alias": "amoeba.noalias all operands",
    }
    operation_hash = _sha256(operation)
    if a_dtype == "i32":
        c_initialization = "caller supplied, read before each first accumulation"
    else:
        c_initialization = "caller supplied, read before each first accumulation"
    return {
        "schema": PROGRAM_SCHEMA,
        "function": function,
        "source_program_sha256": _sha256(source.encode("utf-8")),
        "operation_access_sha256": operation_hash,
        "operation": operation,
        "M": m,
        "N": n,
        "K": k,
        "dtype": a_dtype,
        "C_initialization": c_initialization,
        "layout": "row_major_identity",
        "alias_proof": alias_proof,
    }


def _normalise_program(program: Any) -> dict[str, Any]:
    path = _as_path(program)
    if path is not None:
        raw: Mapping[str, Any] = _parse_canonical_mlir(path)
    elif isinstance(program, Mapping):
        raw = program
        # Accept a descriptor nested under the identity key used by manifests.
        if isinstance(raw.get("program_identity"), Mapping):
            raw = raw["program_identity"]  # type: ignore[assignment]
        elif isinstance(raw.get("descriptor"), Mapping):
            raw = raw["descriptor"]  # type: ignore[assignment]
    else:
        raise VariantError("program must be a source path or descriptor object")

    fields = {"M", "N", "K", "dtype", "layout", "C_initialization",
              "source_program_sha256", "operation_access_sha256",
              "function", "alias_proof"}
    missing = sorted(field for field in fields if field not in raw)
    # Minimal descriptors are useful in unit tests and are still bound to a
    # deterministic source identity.  A source hash is synthesized only when
    # the caller supplied all semantic fields; unknown access/alias information
    # remains a hard failure below.
    if missing:
        allowed_synthesis = {"source_program_sha256", "operation_access_sha256"}
        if set(missing).difference(allowed_synthesis):
            raise VariantError(f"program descriptor is missing {missing!r}")

    m = _positive_int(raw.get("M"), "program.M")
    n = _positive_int(raw.get("N"), "program.N")
    k = _positive_int(raw.get("K"), "program.K")
    dtype = raw.get("dtype")
    if not isinstance(dtype, str) or not dtype or any(ch.isspace() for ch in dtype):
        raise VariantError("program.dtype must be a non-empty type name")
    layout = raw.get("layout")
    if layout == "row_major":
        layout = "row_major_identity"
    if layout != "row_major_identity":
        raise VariantError("unsupported or unknown GEMM layout")
    function = raw.get("function")
    if not isinstance(function, str) or not function:
        raise VariantError("program.function must be a non-empty name")
    c_initialization = raw.get("C_initialization")
    if not isinstance(c_initialization, str) or not c_initialization:
        raise VariantError("program.C_initialization is unknown")
    if "caller" not in c_initialization.lower() or "read" not in c_initialization.lower():
        raise VariantError("C initialization must be caller supplied and read")
    alias_proof = raw.get("alias_proof")
    if alias_proof is True:
        alias_proof = "explicit noalias"
    if (not isinstance(alias_proof, str) or not alias_proof or
            "alias" not in alias_proof.lower() or
            "noalias" not in alias_proof.lower()):
        raise VariantError("program alias proof is not an explicit noalias proof")

    operation = raw.get("operation")
    if operation is None:
        operation = _program_operation(dtype, alias_proof)
    if not isinstance(operation, Mapping):
        raise VariantError("program.operation must be an object")
    expected_access = _program_operation(dtype, alias_proof)["access"]
    if operation.get("access") != expected_access:
        raise VariantError("program access relation is unknown or inconsistent")
    if operation.get("loop_order") != ["M", "N", "K"]:
        raise VariantError("program loop order is unknown or inconsistent")
    operation_alias = operation.get("alias")
    if (not isinstance(operation_alias, str) or
            "noalias" not in operation_alias.lower()):
        raise VariantError("program operation alias proof is inconsistent")

    op_hash = raw.get("operation_access_sha256")
    if op_hash is None:
        op_hash = _sha256(operation)
    if not _is_sha256(op_hash):
        raise VariantError("program.operation_access_sha256 is invalid")
    if raw.get("operation") is not None and op_hash != _sha256(operation):
        raise VariantError("program operation hash does not match operation")
    source_hash = raw.get("source_program_sha256")
    if source_hash is None:
        source_hash = _sha256({
            "function": function, "M": m, "N": n, "K": k,
            "dtype": dtype, "layout": layout,
            "C_initialization": c_initialization,
            "alias_proof": alias_proof, "operation": operation,
        })
    if not _is_sha256(source_hash):
        raise VariantError("program.source_program_sha256 is invalid")
    normalized = {
        "schema": PROGRAM_SCHEMA,
        "function": function,
        "source_program_sha256": source_hash,
        "operation_access_sha256": op_hash,
        "operation": json.loads(_canonical(operation)),
        "M": m,
        "N": n,
        "K": k,
        "dtype": dtype,
        "C_initialization": c_initialization,
        "layout": layout,
        "alias_proof": alias_proof,
    }
    identity_payload = {
        key: normalized[key] for key in (
            "source_program_sha256", "function", "operation_access_sha256",
            "M", "N", "K", "dtype", "layout", "C_initialization",
            "alias_proof")
    }
    normalized["canonical_program_id"] = _sha256(identity_payload)
    return normalized


def canonical_program_identity(program: Any) -> dict[str, Any]:
    """Return the source-bound canonical GEMM identity.

    The returned object contains the source and normalized access hashes used
    by every variant identity.  It is a copy and can safely be modified by a
    caller without changing future enumeration results.
    """

    return _normalise_program(program)


def recognize(program: Any) -> dict[str, Any]:
    """Compatibility spelling for callers of the existing GEMM recognizer."""

    return canonical_program_identity(program)


def canonical_program_id(program: Any) -> str:
    return canonical_program_identity(program)["canonical_program_id"]


def _range_region(buffer: str, lower: Sequence[int], upper: Sequence[int],
                  **extra: Any) -> dict[str, Any]:
    # ``lower``/``upper`` are the same half-open coordinates consumed by the
    # TaskEdgeContract graph-facts adapter.  That adapter may flatten an edge
    # payload to ``region_lower``/``region_upper`` for MLIR attributes; the
    # semantic record keeps one canonical region spelling here.
    if len(lower) != len(upper) or any(a < 0 or b < a
                                       for a, b in zip(lower, upper)):
        raise VariantError("invalid half-open region")
    region = {
        "schema": REGION_SCHEMA,
        "buffer": buffer,
        "lower": [int(value) for value in lower],
        "upper": [int(value) for value in upper],
        "half_open": True,
    }
    region.update(extra)
    return region


def _tiles(extent: int, factor: int, tail_policy: str) -> list[tuple[int, int]]:
    result = []
    start = 0
    while start < extent:
        end = min(start + factor, extent)
        if end - start != factor and tail_policy == "reject":
            return []
        result.append((start, end))
        start = end
    return result


def _edge(source: str, target: str, kind: str, scope: str,
          region: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # ``scope`` and the typed ``kind`` names intentionally match graph-facts
    # edge concepts.  A region is optional for control-only completion edges.
    result: dict[str, Any] = {
        "source_task": source,
        "target_task": target,
        "kind": kind,
        "type": kind,
        "scope": scope,
        "edge_scope": scope,
    }
    if region is not None:
        result["region"] = json.loads(_canonical(region))
    return result


def _task(task_id: str, tile_id: str, k_id: str | None,
          m_range: Sequence[int], n_range: Sequence[int],
          k_range: Sequence[int] | None, regions: Mapping[str, Any],
          policy: str, kind: str = "gemm", semantic_task: bool = True,
          predecessors: Iterable[str] = (), successors: Iterable[str] = (),
          **extra: Any) -> dict[str, Any]:
    region_values = json.loads(_canonical(regions))
    result: dict[str, Any] = {
        "task_id": task_id,
        "task_kind": kind,
        "semantic_task": semantic_task,
        "logical_output_tile_id": tile_id,
        "logical_k_block_id": k_id,
        "m_range": list(m_range),
        "n_range": list(n_range),
        "k_range": list(k_range) if k_range is not None else None,
        "M_range": list(m_range),
        "N_range": list(n_range),
        "K_range": list(k_range) if k_range is not None else None,
        "regions": region_values,
        # Top-level names keep the format easy to consume from graph-facts
        # code while ``regions`` is the canonical grouped form.
        "A_read_region": region_values.get("A_read"),
        "B_read_region": region_values.get("B_read"),
        "C_input_region": region_values.get("C_input"),
        "C_output_region": region_values.get("C_output"),
        "partial_C_region": region_values.get("partial_C"),
        "partial_C_regions": region_values.get("partial_Cs", []),
        "predecessors": sorted(set(predecessors)),
        "successors": sorted(set(successors)),
        "reduction_policy": policy,
    }
    result.update(extra)
    return result


def _variant_identity_payload(variant: Mapping[str, Any]) -> dict[str, Any]:
    """Select semantic fields; backend facts never affect a variant ID."""

    return {
        "schema": SCHEMA,
        "program_identity": variant.get("program_identity"),
        "BM": variant.get("BM"),
        "BN": variant.get("BN"),
        "BK": variant.get("BK"),
        "tail_policy": variant.get("tail_policy"),
        "k_policy": variant.get("k_policy"),
        "reduction_topology": variant.get("reduction_topology"),
        "arithmetic_semantics": variant.get("arithmetic_semantics"),
        "output_tiles": variant.get("output_tiles"),
        "k_blocks": variant.get("k_blocks"),
        "tasks": variant.get("tasks"),
        "edges": variant.get("edges"),
        "completion_task_id": variant.get("completion_task_id"),
        "reduction_structure": variant.get("reduction_structure"),
    }


def _make_variant(program: Mapping[str, Any], bm: int, bn: int, bk: int,
                  policy: str, tail_policy: str,
                  reduction_topology: str) -> dict[str, Any] | None:
    if policy == "sequential_accumulate" and reduction_topology != "none":
        return None
    if policy == "parallel_partial_reduce" and reduction_topology not in (
            "linear", "balanced_tree"):
        return None
    # Any parallel partial reduction changes the original floating-point
    # accumulation grouping, including a linear chain (the partial products
    # are summed before the caller-supplied C value is merged).  Without an
    # explicit reassociation/fast-math contract it is not a legal semantic
    # variant, so source-only f32 enumeration retains sequential tiling only.
    if (program["dtype"] != "i32" and
            policy == "parallel_partial_reduce"):
        return None
    m, n, k = (program[key] for key in ("M", "N", "K"))
    m_tiles = _tiles(m, bm, tail_policy)
    n_tiles = _tiles(n, bn, tail_policy)
    k_blocks = _tiles(k, bk, tail_policy)
    if not m_tiles or not n_tiles or not k_blocks:
        return None
    output_tiles = [
        {
            "logical_output_tile_id": f"tile_m{m0}_n{n0}",
            "m_range": [m0, m1],
            "n_range": [n0, n1],
            "C_region": _range_region("C", [m0, n0], [m1, n1],
                                       access="read_write"),
        }
        for m0, m1 in m_tiles for n0, n1 in n_tiles
    ]
    k_records = [
        {"logical_k_block_id": f"k_{index}_{k0}_{k1}",
         "index": index, "k_range": [k0, k1]}
        for index, (k0, k1) in enumerate(k_blocks)
    ]

    tasks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    final_task_ids: list[str] = []
    reduction_task_ids: list[str] = []
    for tile in output_tiles:
        tile_id = tile["logical_output_tile_id"]
        m0, m1 = tile["m_range"]
        n0, n1 = tile["n_range"]
        c_region = tile["C_region"]
        compute_ids: list[str] = []
        for block in k_records:
            index = block["index"]
            k0, k1 = block["k_range"]
            k_id = block["logical_k_block_id"]
            task_id = f"gemm_{tile_id}_{k_id}"
            compute_ids.append(task_id)
            if policy == "sequential_accumulate":
                c_input = json.loads(_canonical(c_region))
                c_input["access"] = "read_modify_write_state"
                if index == 0:
                    c_owner = "external_initial_C"
                    c_input["ownership"] = "initial_C_once"
                else:
                    c_owner = "state_carry"
                    c_input["ownership"] = "previous_K_block"
                partial = None
            else:
                c_input = None
                c_owner = "reduction_task"
                partial = _range_region(
                    f"partial_C/{tile_id}/{k_id}", [m0, n0], [m1, n1],
                    access="private_write", logical_tile=tile_id,
                    logical_k_block=k_id)
            regions = {
                "A_read": _range_region("A", [m0, k0], [m1, k1],
                                         access="read"),
                "B_read": _range_region("B", [k0, n0], [k1, n1],
                                         access="read"),
                "C_input": c_input,
                # Parallel K tasks publish only their private partial.  The
                # reduction task below is the sole owner of final C output.
                "C_output": (json.loads(_canonical(c_region))
                              if policy == "sequential_accumulate" else None),
                "partial_C": partial,
            }
            tasks.append(_task(
                task_id, tile_id, k_id, [m0, m1], [n0, n1], [k0, k1],
                regions, policy, predecessors=(), successors=(),
                reduction_topology=reduction_topology,
                initial_C_ownership=c_owner,
                final_C_ownership=("external_final_C" if index == len(k_records)-1
                                   and policy == "sequential_accumulate" else
                                   "private_partial_C"),
                arithmetic_order=["m_ascending", "n_ascending", "k_ascending"],
            ))
        if policy == "sequential_accumulate":
            for previous, current in zip(compute_ids, compute_ids[1:]):
                source_task = next(task for task in tasks
                                   if task["task_id"] == previous)
                edges.append(_edge(
                    previous, current, "state_carry", "tile_local",
                    source_task["C_output_region"]))
            final_task_ids.append(compute_ids[-1])
        else:
            compute_by_id = {task["task_id"]: task for task in tasks}
            # A reduction node carries the direct operand IDs and the complete
            # ordered leaf list.  ``region`` is the value produced by the
            # node, which gives every private state a stable lifetime.
            nodes: list[dict[str, Any]] = [
                {
                    "id": task_id,
                    "region": compute_by_id[task_id]["partial_C_region"],
                    "leaves": [task_id],
                    "k0": block["k_range"][0],
                    "k1": block["k_range"][1],
                }
                for task_id, block in zip(compute_ids, k_records)
            ]
            specs: list[dict[str, Any]] = []
            if reduction_topology == "linear":
                previous: dict[str, Any] | None = None
                for index, node in enumerate(nodes):
                    reduction_id = (
                        f"reduce_linear_{tile_id}_step{index}")
                    direct = ([node] if previous is None
                              else [previous, node])
                    output_is_final = index == len(nodes) - 1
                    output = (json.loads(_canonical(c_region))
                              if output_is_final else _range_region(
                                  f"reduction_C/{tile_id}/linear/step{index}",
                                  [m0, n0], [m1, n1],
                                  access="private_reduction_state",
                                  logical_tile=tile_id,
                                  reduction_step=index))
                    specs.append({
                        "task_id": reduction_id,
                        "direct": direct,
                        "leaves": ([leaf for item in direct
                                    for leaf in item["leaves"]]),
                        "k_range": [0, node["k1"]],
                        "output": output,
                        "input": (json.loads(_canonical(c_region))
                                   if previous is None else previous["region"]),
                        "initial": previous is None,
                        "level": 0,
                        "node_index": index,
                    })
                    previous = {
                        "id": reduction_id,
                        "region": output,
                        "leaves": specs[-1]["leaves"],
                        "k0": 0,
                        "k1": node["k1"],
                    }
            else:
                # Pair adjacent nodes at each level.  An odd node is carried
                # unchanged to the next level, giving a deterministic,
                # height-balanced tree for non-power-of-two K block counts.
                level = 0
                while len(nodes) > 1:
                    next_nodes: list[dict[str, Any]] = []
                    for pair_index in range(0, len(nodes), 2):
                        if pair_index + 1 == len(nodes):
                            next_nodes.append(nodes[pair_index])
                            continue
                        left, right = nodes[pair_index], nodes[pair_index + 1]
                        reduction_id = (
                            f"reduce_tree_{tile_id}_l{level}_n{pair_index // 2}")
                        # The root alone consumes initial C and publishes C.
                        # Determine root status from the pair's eventual level
                        # after constructing the next node list below.
                        next_nodes.append({
                            "id": reduction_id,
                            "region": None,
                            "leaves": left["leaves"] + right["leaves"],
                            "k0": left["k0"],
                            "k1": right["k1"],
                        })
                        specs.append({
                            "task_id": reduction_id,
                            "direct": [left, right],
                            "leaves": left["leaves"] + right["leaves"],
                            "k_range": [left["k0"], right["k1"]],
                            "output": None,
                            "input": None,
                            "initial": False,
                            "level": level,
                            "node_index": pair_index // 2,
                            "result_node": next_nodes[-1],
                        })
                    nodes = next_nodes
                    level += 1
                # One K block still needs a reduction task so that initial C
                # ownership and final publication are explicit.
                if len(k_records) == 1 and not specs:
                    node = nodes[0]
                    specs.append({
                        "task_id": f"reduce_tree_{tile_id}_l0_n0",
                        "direct": [node],
                        "leaves": list(node["leaves"]),
                        "k_range": [0, k],
                        "output": None,
                        "input": None,
                        "initial": True,
                        "level": 0,
                        "node_index": 0,
                        "result_node": node,
                    })
                root_id = (nodes[0]["id"] if len(k_records) > 1
                            else specs[0]["task_id"])
                for spec in specs:
                    if spec["task_id"] == root_id:
                        spec["initial"] = True
                        spec["output"] = json.loads(_canonical(c_region))
                        spec["input"] = json.loads(_canonical(c_region))
                    elif spec["output"] is None:
                        step = spec["level"]
                        spec["output"] = _range_region(
                            f"reduction_C/{tile_id}/balanced/l{step}"
                            f"/n{spec['node_index']}", [m0, n0], [m1, n1],
                            access="private_reduction_state",
                            logical_tile=tile_id,
                            reduction_level=step,
                            reduction_node=spec["node_index"])
                    if spec["input"] is None and not spec["initial"]:
                        spec["input"] = None

            # Populate node outputs and reduction tasks in construction order.
            for spec in specs:
                direct = spec["direct"]
                partial_regions = [item["region"] for item in direct]
                # A tree's first level consumes private partials; later levels
                # consume reduction state regions.  The root adds C exactly
                # once through C_input_region.
                c_input = spec["input"]
                if c_input is not None:
                    c_input = json.loads(_canonical(c_input))
                    c_input.setdefault("access", "read_modify_write_state")
                    c_input["ownership"] = (
                        "initial_C_once" if spec["initial"] else "previous_reduction_state")
                output = spec["output"]
                if output is None:
                    raise VariantError("reduction output was not materialized")
                output = json.loads(_canonical(output))
                if "result_node" in spec:
                    spec["result_node"]["region"] = output
                output_is_final = output.get("buffer") == "C"
                reduction_regions = {
                    "A_read": None,
                    "B_read": None,
                    "C_input": c_input,
                    "C_output": output,
                    "partial_C": None,
                    "partial_Cs": partial_regions,
                }
                direct_ids = [item["id"] for item in direct]
                reduction_task = _task(
                    spec["task_id"], tile_id,
                    (f"reduce_{reduction_topology}_{spec['level']}"
                     f"_{spec['node_index']}"),
                    [m0, m1], [n0, n1], spec["k_range"],
                    reduction_regions, policy, kind="reduction",
                    predecessors=direct_ids, successors=(),
                    initial_C_ownership=("reduction_task_once"
                                         if spec["initial"] else "state_carry"),
                    final_C_ownership=("external_final_C" if output_is_final
                                       else "private_reduction_state"),
                    reduction_topology=reduction_topology,
                    reduction_input_task_ids=direct_ids,
                    reduction_input_regions=partial_regions,
                    reduction_order=list(spec["leaves"]),
                    reduction_associativity="ordered_modulo_2^32",
                    reduction_level=spec["level"],
                    reduction_node_index=spec["node_index"],
                    state_lifetime=(None if output_is_final else {
                        "buffer": output["buffer"],
                        "produced_by": spec["task_id"],
                        "last_consumer": None,
                    }),
                )
                tasks.append(reduction_task)
                reduction_task_ids.append(spec["task_id"])
                for item, region in zip(direct, partial_regions):
                    source = item["id"]
                    kind = ("reduction_input" if source.startswith("gemm_")
                            else "reduction_input")
                    edges.append(_edge(source, spec["task_id"], kind,
                                       "tile_local", region))
                # Linear carry is represented as state_carry; tree edges stay
                # reduction_input because each parent combines two operands.
                if reduction_topology == "linear" and len(direct) == 2:
                    edges[-2] = _edge(
                        direct[0]["id"], spec["task_id"], "state_carry",
                        "tile_local", direct[0]["region"])
            # Bind the final consumer of every private state/partial.  This is
            # part of the semantic task record and therefore of the ID.
            task_map = {task["task_id"]: task for task in tasks}
            for spec in specs:
                consumer = spec["task_id"]
                for item in spec["direct"]:
                    source = item["id"]
                    if source.startswith("gemm_"):
                        task_map[source]["partial_lifetime"] = {
                            "buffer": task_map[source]["partial_C_region"]["buffer"],
                            "produced_by": source,
                            "last_consumer": consumer,
                            "release_after": consumer,
                        }
                    elif source in task_map and task_map[source].get("state_lifetime"):
                        task_map[source]["state_lifetime"]["last_consumer"] = consumer
            root_id = specs[-1]["task_id"]
            # For a balanced tree specs are level-order, so identify the task
            # that owns C rather than relying on list position.
            roots = [spec["task_id"] for spec in specs
                     if spec["output"].get("buffer") == "C"]
            final_task_ids.append(roots[0] if roots else root_id)

    completion_id = "completion_join"
    completion_task = _task(
        completion_id, "all_output_tiles", None, [0, m], [0, n], None,
        {"A_read": None, "B_read": None, "C_input": None,
         "C_output": None, "partial_C": None},
        policy, kind="completion_join", semantic_task=False,
        predecessors=final_task_ids, successors=(),
        join_kind="completion_only", initial_C_ownership="none",
        final_C_ownership="none")
    tasks.append(completion_task)
    for final_id in final_task_ids:
        edges.append(_edge(final_id, completion_id, "completion_only",
                           "tensor_wide"))

    # Fill predecessor/successor lists from the typed edge set.  This is kept
    # as a separate deterministic pass so traversal order cannot affect them.
    incoming: dict[str, set[str]] = {task["task_id"]: set() for task in tasks}
    outgoing: dict[str, set[str]] = {task["task_id"]: set() for task in tasks}
    for edge in edges:
        incoming[edge["target_task"]].add(edge["source_task"])
        outgoing[edge["source_task"]].add(edge["target_task"])
    for task in tasks:
        task["predecessors"] = sorted(incoming[task["task_id"]])
        task["successors"] = sorted(outgoing[task["task_id"]])

    variant: dict[str, Any] = {
        "schema": SCHEMA,
        "program_identity": json.loads(_canonical(program)),
        "canonical_program_id": program["canonical_program_id"],
        "BM": bm,
        "BN": bn,
        "BK": bk,
        "tail_policy": tail_policy,
        "tail_required": any(extent % factor != 0 for extent, factor in
                             ((m, bm), (n, bn), (k, bk))),
        "semantic_tail_supported": True,
        "backend_tail_supported": "unknown",
        "k_policy": policy,
        "reduction_topology": reduction_topology,
        "arithmetic_semantics": (
            ARITHMETIC_SEMANTICS if program["dtype"] == "i32"
            else f"{program['dtype']}_semantic"),
        "semantic_legal": True,
        "backend_status": "unknown",
        "output_tiles": output_tiles,
        "k_blocks": k_records,
        "tasks": tasks,
        "edges": edges,
        "completion_task_id": completion_id,
        "reduction_structure": {
            "topology": reduction_topology,
            "root_task_ids": sorted(final_task_ids),
            "reduction_task_count": len(reduction_task_ids),
        },
        "coverage": {
            "M": [0, m], "N": [0, n], "K": [0, k],
            "output_tile_count": len(output_tiles),
            "k_block_count": len(k_records),
        },
    }
    variant_id = _sha256(_variant_identity_payload(variant))
    variant["variant_id"] = variant_id
    variant["semantic_variant_id"] = variant_id
    return variant


def enumerate_variants(program: Any, bm_set: Iterable[int], bn_set: Iterable[int],
                       bk_set: Iterable[int], k_policies: Iterable[str],
                       tail_policy: str = "reject",
                       limit: int | None = None,
                       reduction_topologies: Iterable[str] | str | None = None,
                       reduction_topology: str | None = None) -> list[dict[str, Any]]:
    """Enumerate legal semantic GEMM variants in deterministic order.

    ``tail_policy='reject'`` omits factor combinations that do not exactly
    divide an extent.  ``tail_policy='exact'`` emits a final smaller range and
    validates it like every other range.  By default sequential accumulation
    uses topology ``none`` and parallel reduction expands to both ``linear``
    and ``balanced_tree``.  ``reduction_topology`` is a singular convenience
    spelling; callers may use ``reduction_topologies`` for a set.  The optional
    limit is applied only after sorting and deduplication.
    """

    descriptor = _normalise_program(program)
    bms = _normalise_factor_set(bm_set, "bm_set")
    bns = _normalise_factor_set(bn_set, "bn_set")
    bks = _normalise_factor_set(bk_set, "bk_set")
    policies = _normalise_policy_set(k_policies)
    tail = _normalise_tail_policy(tail_policy)
    # Preserve a convenient positional spelling introduced with the topology
    # dimension while retaining the established seventh ``limit`` argument.
    if isinstance(limit, str) and reduction_topologies is None and reduction_topology is None:
        reduction_topology = limit
        limit = None
    if reduction_topology is not None:
        if reduction_topologies is not None:
            raise VariantError(
                "specify reduction_topology or reduction_topologies, not both")
        reduction_topologies = reduction_topology
    topologies = _normalise_topology_set(reduction_topologies, policies)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise VariantError("limit must be a positive integer or None")
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bm in bms:
        for bn in bns:
            for bk in bks:
                for policy in policies:
                    policy_topologies = ("none",) if policy == "sequential_accumulate" else tuple(
                        topology for topology in topologies
                        if topology in PARALLEL_REDUCTION_TOPOLOGIES)
                    for topology in policy_topologies:
                        # A factor larger than a static extent covers that
                        # entire axis once.  Canonicalize it to the identity
                        # factor so exact-tail requests cannot publish a
                        # second ID for the same task graph.
                        effective = ((min(bm, descriptor["M"]),
                                      min(bn, descriptor["N"]),
                                      min(bk, descriptor["K"]))
                                     if tail == "exact" else (bm, bn, bk))
                        variant = _make_variant(
                            descriptor, *effective, policy, tail, topology)
                        if variant is None or variant["variant_id"] in seen:
                            continue
                        seen.add(variant["variant_id"])
                        variants.append(variant)
    variants.sort(key=lambda item: (
        item["BM"], item["BN"], item["BK"],
        SUPPORTED_K_POLICIES.index(item["k_policy"]),
        SUPPORTED_REDUCTION_TOPOLOGIES.index(item["reduction_topology"]),
        item["variant_id"]))
    if limit is not None:
        variants = variants[:limit]
    return variants


def enumeration_statistics(program: Any, bm_set: Iterable[int],
                           bn_set: Iterable[int], bk_set: Iterable[int],
                           k_policies: Iterable[str],
                           tail_policy: str = "reject",
                           reduction_topologies: Iterable[str] | str | None = None
                           ) -> dict[str, Any]:
    """Classify every requested semantic combination deterministically."""

    descriptor = _normalise_program(program)
    bms = _normalise_factor_set(bm_set, "bm_set")
    bns = _normalise_factor_set(bn_set, "bn_set")
    bks = _normalise_factor_set(bk_set, "bk_set")
    policies = _normalise_policy_set(k_policies)
    tail = _normalise_tail_policy(tail_policy)
    topologies = _normalise_topology_set(reduction_topologies, policies)
    histogram: dict[str, int] = {}
    requested = 0
    for bm in bms:
        for bn in bns:
            for bk in bks:
                for policy in policies:
                    applicable = (("none",) if policy == "sequential_accumulate"
                                  else tuple(topology for topology in topologies
                                             if topology in PARALLEL_REDUCTION_TOPOLOGIES))
                    for _topology in applicable:
                        requested += 1
                        reason = None
                        if tail == "reject" and any(
                                extent % factor != 0 for extent, factor in
                                ((descriptor["M"], bm),
                                 (descriptor["N"], bn),
                                 (descriptor["K"], bk))):
                            reason = "nondivisible_static_tail"
                        elif (descriptor["dtype"] != "i32" and
                              policy == "parallel_partial_reduce"):
                            reason = "f32_reassociation_not_permitted"
                        if reason is not None:
                            histogram[reason] = histogram.get(reason, 0) + 1
    return {"requested_combination_count": requested,
            "rejected_count": sum(histogram.values()),
            "legal_raw_combination_count": requested - sum(histogram.values()),
            "rejection_reason_histogram": dict(sorted(histogram.items()))}


def _cells(region: Mapping[str, Any]) -> set[tuple[int, ...]]:
    lower, upper = region["lower"], region["upper"]
    if len(lower) != len(upper):
        raise VariantValidationError("region rank mismatch")
    result: set[tuple[int, ...]] = {()}
    for lo, hi in zip(lower, upper):
        result = {prefix + (index,) for prefix in result
                  for index in range(lo, hi)}
    return result


def _validate_region(region: Any, expected_buffer: str | None = None) -> None:
    if region is None:
        return
    if not isinstance(region, Mapping) or region.get("schema") != REGION_SCHEMA:
        raise VariantValidationError("malformed region")
    if expected_buffer is not None and region.get("buffer") != expected_buffer:
        raise VariantValidationError("region buffer mismatch")
    lower, upper = region.get("lower"), region.get("upper")
    if (not isinstance(lower, list) or not isinstance(upper, list) or
            len(lower) != len(upper) or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in lower + upper) or
            any(lo > hi for lo, hi in zip(lower, upper)) or
            region.get("half_open") is not True):
        raise VariantValidationError("malformed half-open region")


def _validate_graph(variant: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    tasks = variant.get("tasks")
    edges = variant.get("edges")
    if not isinstance(tasks, list) or not tasks:
        raise VariantValidationError("variant tasks are missing")
    if not isinstance(edges, list):
        raise VariantValidationError("variant edges are missing")
    task_map: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise VariantValidationError("task must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in task_map:
            raise VariantValidationError("task IDs must be unique")
        task_map[task_id] = task
        if (task.get("M_range") != task.get("m_range") or
                task.get("N_range") != task.get("n_range") or
                task.get("K_range") != task.get("k_range")):
            raise VariantValidationError("task range aliases disagree")
        for key in ("A_read_region", "B_read_region", "C_input_region",
                    "C_output_region", "partial_C_region"):
            _validate_region(task.get(key))
        regions = task.get("regions")
        if not isinstance(regions, Mapping):
            raise VariantValidationError("task regions are missing")
        for key in ("A_read", "B_read", "C_input", "C_output", "partial_C"):
            if key not in regions:
                raise VariantValidationError("task region field is missing")
            _validate_region(regions.get(key))
        if task.get("A_read_region") != regions.get("A_read"):
            raise VariantValidationError("A region aliases disagree")
        if (task.get("B_read_region") != regions.get("B_read") or
                task.get("C_input_region") != regions.get("C_input") or
                task.get("C_output_region") != regions.get("C_output") or
                task.get("partial_C_region") != regions.get("partial_C")):
            raise VariantValidationError("task region aliases disagree")
        partials = regions.get("partial_Cs", [])
        if not isinstance(partials, list):
            raise VariantValidationError("partial C regions must be an array")
        for partial in partials:
            _validate_region(partial)
        if task.get("partial_C_regions") != partials:
            raise VariantValidationError("partial C aliases disagree")
    adjacency: dict[str, set[str]] = {task_id: set() for task_id in task_map}
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise VariantValidationError("edge must be an object")
        source, target = edge.get("source_task"), edge.get("target_task")
        if source not in task_map or target not in task_map or source == target:
            raise VariantValidationError("edge references an unknown task")
        if edge.get("type") != edge.get("kind") or edge.get("scope") != edge.get("edge_scope"):
            raise VariantValidationError("edge aliases disagree")
        if edge.get("kind") not in ("state_carry", "reduction_input",
                                     "completion_only"):
            raise VariantValidationError("unknown semantic edge kind")
        if edge.get("scope") not in ("tile_local", "tensor_wide"):
            raise VariantValidationError("unknown semantic edge scope")
        _validate_region(edge.get("region"))
        adjacency[source].add(target)
    # Kahn's algorithm catches cycles independently of the generated expected
    # graph and keeps fail-closed validation useful for hand-written variants.
    indegree = {task_id: 0 for task_id in task_map}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(task_map):
        raise VariantValidationError("task graph contains a cycle")
    if variant.get("completion_task_id") not in task_map:
        raise VariantValidationError("completion task is missing")
    completion = task_map[variant["completion_task_id"]]
    if completion.get("task_kind") != "completion_join" or completion.get("semantic_task") is not False:
        raise VariantValidationError("completion task is not a completion-only join")
    if variant.get("k_policy") == "sequential_accumulate":
        if variant.get("reduction_topology") != "none":
            raise VariantValidationError("sequential graph has a reduction topology")
        if any(edge.get("kind") == "reduction_input" for edge in edges):
            raise VariantValidationError("sequential graph contains a reduction")
    elif variant.get("k_policy") == "parallel_partial_reduce":
        if variant.get("reduction_topology") not in PARALLEL_REDUCTION_TOPOLOGIES:
            raise VariantValidationError("parallel graph has an invalid topology")
        if not any(task.get("task_kind") == "reduction" for task in tasks):
            raise VariantValidationError("parallel graph has no reduction task")

    # Output coverage and K coverage are checked as sets of logical indices.
    m, n, k = program["M"], program["N"], program["K"]
    output_tiles = variant.get("output_tiles")
    if not isinstance(output_tiles, list) or not output_tiles:
        raise VariantValidationError("output tiles are missing")
    output_coverage: list[tuple[int, ...]] = []
    for tile in output_tiles:
        if not isinstance(tile, Mapping):
            raise VariantValidationError("malformed output tile")
        m_range, n_range = tile.get("m_range"), tile.get("n_range")
        if (not isinstance(m_range, list) or not isinstance(n_range, list) or
                len(m_range) != 2 or len(n_range) != 2 or
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in m_range + n_range) or
                not (0 <= m_range[0] < m_range[1] <= m) or
                not (0 <= n_range[0] < n_range[1] <= n)):
            raise VariantValidationError("malformed output tile range")
        cr = tile.get("C_region")
        _validate_region(cr, "C")
        if cr is None:
            raise VariantValidationError("output tile C region is missing")
        lower = cr["lower"]
        upper = cr["upper"]
        if lower != [m_range[0], n_range[0]] or upper != [m_range[1], n_range[1]]:
            raise VariantValidationError("output tile C region mismatch")
        output_coverage.extend(_cells(cr))
    expected_output = {(mi, ni) for mi in range(m) for ni in range(n)}
    if set(output_coverage) != expected_output or len(output_coverage) != len(set(output_coverage)):
        raise VariantValidationError("output regions do not exactly cover C")
    for task in tasks:
        if task.get("task_kind") != "gemm":
            continue
        if task.get("k_range") is None:
            raise VariantValidationError("GEMM task has no K range")
        kr = task["k_range"]
        if not (0 <= kr[0] < kr[1] <= k):
            raise VariantValidationError("GEMM task K range is out of bounds")
        ar, br = task["A_read_region"], task["B_read_region"]
        if ar is None or br is None:
            raise VariantValidationError("GEMM task is missing A/B region")
        if ar["lower"] != [task["m_range"][0], kr[0]] or ar["upper"] != [task["m_range"][1], kr[1]]:
            raise VariantValidationError("A region does not match affine access")
        if br["lower"] != [kr[0], task["n_range"][0]] or br["upper"] != [kr[1], task["n_range"][1]]:
            raise VariantValidationError("B region does not match affine access")
        if variant.get("k_policy") == "parallel_partial_reduce":
            if task.get("C_output_region") is not None or task.get("partial_C_region") is None:
                raise VariantValidationError(
                    "parallel GEMM task must write a private partial only")
            if not isinstance(task.get("partial_lifetime"), Mapping):
                raise VariantValidationError("parallel partial lifetime is missing")
        elif task.get("C_output_region") is None:
            raise VariantValidationError("sequential GEMM task must publish C state")

    if variant.get("k_policy") == "parallel_partial_reduce":
        reductions_by_tile: dict[str, list[Mapping[str, Any]]] = {}
        for task in tasks:
            if task.get("task_kind") != "reduction":
                continue
            if task.get("reduction_topology") != variant.get("reduction_topology"):
                raise VariantValidationError("reduction task topology mismatch")
            output = task.get("C_output_region")
            if output is None:
                raise VariantValidationError("reduction task has no output region")
            _validate_region(output)
            tile_id = task.get("logical_output_tile_id")
            reductions_by_tile.setdefault(tile_id, []).append(task)
            if task.get("final_C_ownership") == "external_final_C":
                if output.get("buffer") != "C":
                    raise VariantValidationError("final reduction does not publish C")
            elif output.get("buffer") == "C":
                raise VariantValidationError("intermediate reduction writes final C")
            if task.get("state_lifetime") is not None:
                lifetime = task["state_lifetime"]
                if not isinstance(lifetime, Mapping) or not lifetime.get("last_consumer"):
                    raise VariantValidationError("reduction state lifetime is incomplete")
        for tile_id, tile_reductions in reductions_by_tile.items():
            final_count = sum(task.get("final_C_ownership") == "external_final_C"
                              for task in tile_reductions)
            initial_count = sum(task.get("initial_C_ownership") == "reduction_task_once"
                                for task in tile_reductions)
            if final_count != 1 or initial_count != 1:
                raise VariantValidationError(
                    "each output tile needs one final reduction and one initial C")


def validate_variant(program: Any, variant: Mapping[str, Any]) -> bool:
    """Validate one variant independently of any backend.

    Validation regenerates the semantic graph from the program and factors,
    then checks the graph itself.  This catches both ordinary field mutations
    and a malformed hand-written graph that happens to carry a recomputed ID.
    """

    if not isinstance(variant, Mapping):
        raise VariantValidationError("variant must be an object")
    descriptor = _normalise_program(program)
    for key in ("BM", "BN", "BK", "k_policy", "tail_policy",
                "reduction_topology"):
        if key not in variant:
            raise VariantValidationError(f"variant.{key} is missing")
    if variant["k_policy"] not in SUPPORTED_K_POLICIES:
        raise VariantValidationError("variant K policy is unsupported")
    if variant["reduction_topology"] not in SUPPORTED_REDUCTION_TOPOLOGIES:
        raise VariantValidationError("variant reduction topology is unsupported")
    expected = _make_variant(
        descriptor,
        _positive_int(variant["BM"], "variant.BM"),
        _positive_int(variant["BN"], "variant.BN"),
        _positive_int(variant["BK"], "variant.BK"),
        variant["k_policy"],
        _normalise_tail_policy(variant["tail_policy"]),
        variant["reduction_topology"],
    )
    if expected is None:
        raise VariantValidationError("variant uses a rejected tail policy")
    if variant.get("schema") != SCHEMA:
        raise VariantValidationError("variant schema mismatch")
    if variant.get("program_identity") != expected["program_identity"]:
        raise VariantValidationError("program identity mismatch")
    if variant.get("semantic_legal") is not True:
        raise VariantValidationError("variant is not semantically legal")
    expected_arithmetic = (ARITHMETIC_SEMANTICS if descriptor["dtype"] == "i32"
                           else f"{descriptor['dtype']}_semantic")
    if variant.get("arithmetic_semantics") != expected_arithmetic:
        raise VariantValidationError("arithmetic semantics mismatch")
    if variant.get("variant_id") != expected["variant_id"] or variant.get("semantic_variant_id") != expected["variant_id"]:
        raise VariantValidationError("variant identity mismatch")
    for key in ("BM", "BN", "BK", "tail_policy", "tail_required",
                "semantic_tail_supported", "backend_tail_supported", "k_policy",
                "reduction_topology",
                "output_tiles", "k_blocks", "tasks", "edges",
                "completion_task_id", "coverage", "reduction_structure"):
        if variant.get(key) != expected.get(key):
            raise VariantValidationError(f"variant semantic field mutated: {key}")
    _validate_graph(variant, descriptor)
    return True


def _runtime_vector(values: Mapping[str, Any], name: str, size: int) -> list[int]:
    raw = values.get(name)
    if raw is None and name == "C_initial":
        raw = values.get("C")
    if raw is None:
        raise VariantError(f"runtime value {name} is missing")
    # Flatten a convenient nested row-major form without accepting arbitrary
    # iterables whose traversal could make execution non-deterministic.
    if isinstance(raw, (str, bytes)):
        raise VariantError(f"runtime value {name} must be an integer vector")
    if isinstance(raw, Sequence) and raw and isinstance(raw[0], Sequence) and not isinstance(raw[0], (str, bytes)):
        flattened = [item for row in raw for item in row]
    else:
        flattened = list(raw)
    if len(flattened) != size:
        raise VariantError(f"runtime value {name} has length {len(flattened)}, expected {size}")
    if any(isinstance(item, bool) or not isinstance(item, int) or
           item < -(1 << 31) or item >= (1 << 32) for item in flattened):
        raise VariantError(f"runtime value {name} contains a non-i32 word")
    return [item & 0xffffffff for item in flattened]


def _mul_add(acc: int, a: int, b: int) -> int:
    return (acc + a * b) & 0xffffffff


def reference(program_or_m: Any, *args: Any) -> list[int]:
    """Compute row-major GEMM with explicit i32 modulo-2**32 arithmetic.

    Both ``reference(program, values)`` and the compact legacy spelling
    ``reference(M, N, K, values)`` are accepted.
    """

    if isinstance(program_or_m, int):
        if len(args) != 3:
            raise VariantError("reference(M, N, K, values) expects four arguments")
        m = _positive_int(program_or_m, "M")
        n = _positive_int(args[0], "N")
        k = _positive_int(args[1], "K")
        values = args[2]
        if not isinstance(values, Mapping):
            raise VariantError("reference values must be an object")
    else:
        if len(args) != 1 or not isinstance(args[0], Mapping):
            raise VariantError("reference(program, values) expects two arguments")
        program = _normalise_program(program_or_m)
        if program["dtype"] != "i32":
            raise VariantError("numeric reference currently supports i32 only")
        m, n, k = program["M"], program["N"], program["K"]
        values = args[0]
    a = _runtime_vector(values, "A", m * k)
    b = _runtime_vector(values, "B", k * n)
    out = _runtime_vector(values, "C_initial", m * n)
    for mi in range(m):
        for ni in range(n):
            index = mi * n + ni
            for ki in range(k):
                out[index] = _mul_add(out[index], a[mi * k + ki],
                                       b[ki * n + ni])
    return out


def interpret_variant(program: Any, variant: Mapping[str, Any],
                      values: Mapping[str, Any]) -> list[int]:
    """Interpret a legal semantic graph, without backend assumptions."""

    validate_variant(program, variant)
    descriptor = _normalise_program(program)
    if descriptor["dtype"] != "i32":
        raise VariantError("numeric interpreter currently supports i32 only")
    if not isinstance(values, Mapping):
        raise VariantError("runtime values must be an object")
    m, n, k = descriptor["M"], descriptor["N"], descriptor["K"]
    a = _runtime_vector(values, "A", m * k)
    b = _runtime_vector(values, "B", k * n)
    result = _runtime_vector(values, "C_initial", m * n)
    tasks = variant["tasks"]
    if variant["k_policy"] == "sequential_accumulate":
        for task in tasks:
            if task["task_kind"] != "gemm":
                continue
            m0, m1 = task["m_range"]
            n0, n1 = task["n_range"]
            k0, k1 = task["k_range"]
            for mi in range(m0, m1):
                for ni in range(n0, n1):
                    destination = mi * n + ni
                    for ki in range(k0, k1):
                        result[destination] = _mul_add(
                            result[destination], a[mi * k + ki],
                            b[ki * n + ni])
        return result

    # Partial computation starts from additive zero.  Reduction tasks then
    # consume their explicit direct operands in the recorded order.  A
    # reduction task owns a private state buffer until the one final task per
    # tile publishes C; initial C is used only by that tile's root/first task.
    partials: dict[str, list[int]] = {}
    for task in tasks:
        if task["task_kind"] != "gemm":
            continue
        m0, m1 = task["m_range"]
        n0, n1 = task["n_range"]
        k0, k1 = task["k_range"]
        tile_size = (m1 - m0) * (n1 - n0)
        local = [0] * tile_size
        for mi in range(m0, m1):
            for ni in range(n0, n1):
                local[(mi - m0) * (n1 - n0) + ni - n0] = 0
                for ki in range(k0, k1):
                    local[(mi - m0) * (n1 - n0) + ni - n0] = _mul_add(
                        local[(mi - m0) * (n1 - n0) + ni - n0],
                        a[mi * k + ki], b[ki * n + ni])
        partials[task["task_id"]] = local
    reduction_values: dict[str, list[int]] = {}
    for task in tasks:
        if task["task_kind"] != "reduction":
            continue
        m0, m1 = task["m_range"]
        n0, n1 = task["n_range"]
        width = n1 - n0
        direct_ids = task.get("reduction_input_task_ids")
        if not isinstance(direct_ids, list) or not direct_ids:
            raise VariantError("reduction task has no explicit operands")
        operands: list[list[int]] = []
        for task_id in direct_ids:
            if task_id in partials:
                operands.append(partials[task_id])
            elif task_id in reduction_values:
                operands.append(reduction_values[task_id])
            else:
                raise VariantError(f"reduction operand {task_id} is unavailable")
        initial = task.get("initial_C_ownership") == "reduction_task_once"
        output_local = [0] * ((m1 - m0) * width)
        for mi in range(m0, m1):
            for ni in range(n0, n1):
                local_index = (mi - m0) * width + ni - n0
                if initial:
                    accumulator = result[mi * n + ni]
                    operand_start = 0
                else:
                    # Linear reductions carry their preceding reduction task
                    # as the first direct operand; tree non-roots combine only
                    # private operands.  C is never re-read here.
                    accumulator = operands[0][local_index]
                    operand_start = 1
                for operand in operands[operand_start:]:
                    accumulator = _mul_add(accumulator, operand[local_index], 1)
                output_local[local_index] = accumulator
        if task.get("C_output_region", {}).get("buffer") == "C":
            for mi in range(m0, m1):
                for ni in range(n0, n1):
                    result[mi * n + ni] = output_local[(mi - m0) * width + ni - n0]
        else:
            reduction_values[task["task_id"]] = output_local
    return result


__all__ = [
    "ARITHMETIC_SEMANTICS",
    "PROGRAM_SCHEMA",
    "PARALLEL_REDUCTION_TOPOLOGIES",
    "REGION_SCHEMA",
    "SCHEMA",
    "SUPPORTED_REDUCTION_TOPOLOGIES",
    "VariantError",
    "VariantValidationError",
    "canonical_program_id",
    "canonical_program_identity",
    "enumeration_statistics",
    "enumerate_variants",
    "interpret_variant",
    "recognize",
    "reference",
    "validate_variant",
]
