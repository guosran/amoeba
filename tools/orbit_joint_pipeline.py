#!/usr/bin/env python3
"""Source-bound semantic identity and reference evaluator for one ORBIT pipeline.

The pipeline accepted here is deliberately narrow:

    T = C_initial + A * B
    D = T + E

It is represented by one Taskflow MLIR module containing one producer task and
one consumer task.  The C++ MLIR pass recognizes the production source.  This
independent recognizer checks it without a separately written descriptor and
does not know about a backend, a cost model, mapping, or scheduling.

The parser is intentionally strict.  It is not a general MLIR parser; the
actual source is verified by ``mlir-amoeba-opt`` in the fixture and tests.  A
strict source recognizer is useful here because accepting an unfamiliar access
or effect would silently turn an unproved rewrite into a semantic candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "orbit-canonical-producer-consumer-pipeline-v1"
ARITHMETIC_SEMANTICS = "i32_modulo_2^32"
_MEMREF = r"memref<8x8xi32>"


class PipelineValidationError(ValueError):
    """A source failed a named semantic admission check."""

    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fail(code: str, message: str) -> None:
    raise PipelineValidationError(code, message)


def _source_text(source: Any) -> tuple[str, str | None]:
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8"), str(source)
        except (OSError, UnicodeDecodeError) as error:
            _fail("SOURCE_READ_FAILED", str(error))
    if isinstance(source, str):
        # Newlines identify inline MLIR.  Avoid asking the filesystem about a
        # complete module string: a long module can exceed the platform's
        # single-path limit before ``Path.is_file`` gets to return False.
        if "\n" not in source and "\r" not in source:
            candidate = Path(source)
            if candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8"), str(candidate)
                except (OSError, UnicodeDecodeError) as error:
                    _fail("SOURCE_READ_FAILED", str(error))
        return source, None
    _fail("SOURCE_TYPE_INVALID", "source must be a path or MLIR text")


def _balanced_block(text: str, opening: int) -> tuple[int, int]:
    if opening < 0 or opening >= len(text) or text[opening] != "{":
        _fail("SOURCE_STRUCTURE_INVALID", "missing block opening brace")
    depth = 0
    quote = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return opening, index
            if depth < 0:
                break
    _fail("SOURCE_STRUCTURE_INVALID", "unbalanced braces")


def _split_args(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    result.append(text[start:].strip())
    return [item for item in result if item]


def _names(segment: str) -> list[str]:
    return re.findall(r"%[A-Za-z_][A-Za-z0-9_]*", segment)


def _task_spans(body: str) -> list[tuple[re.Match[str], str]]:
    starts = list(re.finditer(
        r"(?P<result>%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*taskflow\.task\s+"
        r"@(?P<name>[A-Za-z_][A-Za-z0-9_.]*)", body))
    result: list[tuple[re.Match[str], str]] = []
    for match in starts:
        # Optional operation attributes contain braces of their own.  Locate
        # the task region after the function-like result type, not the first
        # brace in the operation header.
        region_start = re.search(r"->\s*\([^)]*\)\s*\{", body[match.end():])
        if region_start is None:
            _fail("TASK_REGION_INVALID", f"task @{match.group('name')} has no body")
        opening = match.end() + region_start.end() - 1
        _, closing = _balanced_block(body, opening)
        result.append((match, body[match.start():closing + 1]))
    return result


def _header_segment(task: str, keyword: str) -> str:
    match = re.search(re.escape(keyword) + r"\((.*?)\)", task, flags=re.S)
    if match is None:
        _fail("TASK_CONTRACT_INVALID", f"task header misses {keyword}")
    return match.group(1)


def _counter_info(task: str, expected_count: int) -> tuple[list[str], list[str], dict[str, int]]:
    constants = {
        name: int(value) for name, value in re.findall(
            r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            r"arith\.constant\s+(\d+)\s*:\s*index\s*$", task)
    }
    if set(constants.values()) != {0, 1, 8}:
        _fail("DYNAMIC_CONTROL", "loop constants are not exactly static 0, 1, 8")
    counters = re.findall(
        r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*taskflow\.counter\s+"
        r"(?:(?:parent\((%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*index\)\s+)?)"
        r"from\s+(%[A-Za-z_][A-Za-z0-9_]*)\s+to\s+"
        r"(%[A-Za-z_][A-Za-z0-9_]*)\s+step\s+(%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*index",
        task)
    if len(counters) != expected_count:
        _fail("REGION_MISMATCH", "unexpected number of iteration counters")
    names: list[str] = []
    parents: list[str] = []
    for index, (name, parent, lower, upper, step) in enumerate(counters):
        if constants.get(lower) != 0 or constants.get(upper) != 8 or constants.get(step) != 1:
            _fail("DYNAMIC_CONTROL", "counter bounds are not static unit-step 0..8")
        if index == 0 and parent:
            _fail("REGION_MISMATCH", "outer counter unexpectedly has a parent")
        if index and parent != names[index - 1]:
            _fail("REGION_MISMATCH", "counter parent chain is not row-major")
        names.append(name)
        parents.append(parent)
    block_match = re.search(r"\}\)\s*:\s*\([^)]*\)\s*->\s*\(\)" , task)
    if block_match is None:
        _fail("TASK_REGION_INVALID", "hyperblock region is missing")
    hyperblock = re.search(
        r'"taskflow\.hyperblock"\(([^)]*)\)\s*<\{', task)
    if hyperblock is None or _names(hyperblock.group(1)) != names:
        _fail("REGION_MISMATCH", "hyperblock indices do not match counters")
    # The task region itself has an entry block, followed by the nested
    # hyperblock entry block.  Select the block belonging to the hyperblock;
    # counting every ``^bb0`` in the task would accidentally reject every
    # valid task (and would make it easy to inspect the wrong region).
    hyperblock_body = re.search(
        r'"taskflow\.hyperblock".*?\^bb0\(([^)]*)\):', task,
        flags=re.S)
    if hyperblock_body is None:
        _fail("TASK_REGION_INVALID", "task must have one hyperblock body")
    body_args = _names(hyperblock_body.group(1))
    if len(body_args) != expected_count:
        _fail("REGION_MISMATCH", "hyperblock body rank is not static")
    return names, body_args, constants


def _validate_task_body(task: str, kind: str, task_result: str,
                        external: Mapping[str, str], t_name: str,
                        producer_result: str | None = None) -> dict[str, Any]:
    # The producer owns M/N iteration and contains one static K reduction
    # inside its hyperblock.  Keeping K as an inner SCF loop makes the
    # accumulator initialization and carry explicit while preserving the
    # canonical task/counter shape consumed by the graph-facts pass.
    expected_count = 2 if kind == "producer" else 2
    if kind == "producer" and len(re.findall(r"\btaskflow\.counter\b", task)) != 2:
        _fail("STATIC_K_REDUCTION_MISSING",
              "producer K must be an inner scf.for, not a taskflow counter")
    counters, body_indices, constants = _counter_info(task, expected_count)
    block_header = re.search(r"\^bb0\(([^)]*)\):", task)
    if block_header is None:
        _fail("TASK_REGION_INVALID", f"{kind} block arguments are missing")
    block_args = _names(block_header.group(1))
    expected_args = 4 if kind == "producer" else 3
    if len(block_args) != expected_args:
        _fail("TASK_CONTRACT_INVALID", f"{kind} task has the wrong body arity")

    # Structural allow-list catches unknown accesses, calls, side effects,
    # dynamic control, and accidental second regions before access matching.
    lines = [line.strip() for line in task.splitlines() if line.strip()]
    ignored = ("^bb0(", "})", "}", "{", "}) :")
    in_task_entry_header = False
    for line in lines:
        if line.startswith("^bb0("):
            in_task_entry_header = not line.endswith("):")
            continue
        if in_task_entry_header:
            if line.endswith("):"):
                in_task_entry_header = False
            continue
        if re.match(r"%[A-Za-z_][A-Za-z0-9_]*\s*=\s*taskflow\.task\b", line):
            continue
        if (line.startswith("%") and ("= memref.load " in line or
                                       "= arith." in line or
                                       "= taskflow.counter " in line or
                                       "= arith.constant " in line or
                                       "= scf.for " in line)):
            continue
        if line.startswith('"taskflow.hyperblock"') or \
                line.startswith("taskflow.hyperblock.yield") or \
                line.startswith("taskflow.yield done_writes(") or \
                line.startswith("^bb0(") or line in ignored:
            continue
        if line.startswith("scf.yield ") or line.startswith("iter_args("):
            continue
        if line.startswith(("call ", "func.call ", "cf.", "test.")):
            _fail("UNKNOWN_ACCESS_OR_EFFECT",
                  f"unsupported side effect or control operation in {kind}: {line}")
        if line.startswith("memref.store ") or line.startswith("memref.copy "):
            continue
        if line.startswith("%") or line.startswith("taskflow.") or \
                line.startswith("memref.") or line.startswith("arith."):
            _fail("UNKNOWN_ACCESS_OR_EFFECT", f"unsupported operation in {kind}: {line}")

    if task.count('"taskflow.hyperblock"') != 1:
        _fail("UNKNOWN_ACCESS_OR_EFFECT", f"{kind} has multiple or no hyperblocks")
    if task.count("taskflow.hyperblock.yield") != 1:
        _fail("TASK_REGION_INVALID", f"{kind} hyperblock yield is malformed")
    if task.count("taskflow.yield done_writes") != 1:
        _fail("TASK_REGION_INVALID", f"{kind} task yield is malformed")

    loads = re.findall(
        r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*memref\.load\s+"
        r"(%[A-Za-z_][A-Za-z0-9_]*)\[([^]]+)\]\s*:\s*memref<([^>]+)>\s*$",
        task)
    stores = re.findall(
        r"(?m)^\s*memref\.store\s+(%[A-Za-z_][A-Za-z0-9_]*),\s*"
        r"(%[A-Za-z_][A-Za-z0-9_]*)\[([^]]+)\]\s*:\s*memref<([^>]+)>\s*$",
        task)
    if kind == "producer":
        if len(loads) != 3 or len(stores) != 1:
            _fail("ACCESS_RELATION_UNKNOWN", "producer access count is not GEMM")
        arith_ops = re.findall(r"\barith\.([A-Za-z_][A-Za-z0-9_]*)\b", task)
        if any(op not in {"constant", "muli", "addi"} for op in arith_ops):
            _fail("UNKNOWN_ACCESS_OR_EFFECT", "producer contains unsupported arithmetic")
        if re.search(r"(?m)^\s*memref\.copy\b", task):
            _fail("UNKNOWN_ACCESS_OR_EFFECT",
                  "producer cannot initialize T with an untracked copy")
        scf_ops = re.findall(r"\bscf\.([A-Za-z_][A-Za-z0-9_]*)", task)
        if any(op not in {"for", "yield"} for op in scf_ops):
            _fail("UNKNOWN_ACCESS_OR_EFFECT", "producer contains unsupported scf operation")
        if scf_ops.count("for") != 1 or scf_ops.count("yield") != 1:
            _fail("STATIC_K_REDUCTION_MISSING",
                  "producer must contain exactly one static scf.for reduction")
        reduction = re.search(
            r"(?ms)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*scf\.for\s+"
            r"(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(%[A-Za-z_][A-Za-z0-9_]*)\s+"
            r"to\s+(%[A-Za-z_][A-Za-z0-9_]*)\s+step\s+"
            r"(%[A-Za-z_][A-Za-z0-9_]*)\s+iter_args\(\s*"
            r"(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*"
            r"->\s*\(i32\)\s*\{(.*?)^\s*\}", task)
        if reduction is None:
            _fail("STATIC_K_REDUCTION_MISSING",
                  "producer K reduction has no canonical static scf.for form")
        reduction_result, k_index, lower, upper, step, accumulator, initial, body = reduction.groups()
        if constants.get(lower) != 0 or constants.get(upper) != 8 or constants.get(step) != 1:
            _fail("DYNAMIC_CONTROL", "producer K reduction bounds are not static unit-step 0..8")
        if not re.search(
                rf"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*memref\.load\s+"
                rf"{re.escape(block_args[2])}\[{re.escape(body_indices[0])},\s*"
                rf"{re.escape(body_indices[1])}\]\s*:\s*memref<8x8xi32>\s*$",
                task):
            _fail("INITIAL_STATE_NOT_ESTABLISHED",
                  "producer accumulator is not initialized from C_initial")
        expected_loads = [
            (block_args[2], (body_indices[0], body_indices[1])),
            (block_args[0], (body_indices[0], k_index)),
            (block_args[1], (k_index, body_indices[1])),
        ]
        for actual, expected in zip(loads, expected_loads):
            value, buffer, indices, memref_type = actual
            if (buffer, tuple(item.strip() for item in indices.split(",")),
                    memref_type) != (expected[0], expected[1], "8x8xi32"):
                _fail("ACCESS_RELATION_UNKNOWN", "producer access is not C[M,N], A[M,K], B[K,N]")
        product = re.search(
            r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*arith\.muli\s+"
            r"(%[A-Za-z_][A-Za-z0-9_]*),\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*i32\s*$", body)
        updated = re.search(
            r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*arith\.addi\s+"
            r"(%[A-Za-z_][A-Za-z0-9_]*),\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*i32\s*$", body)
        if product is None or updated is None or len(re.findall(r"arith\.addi", task)) != 1:
            _fail("UNKNOWN_ACCESS_OR_EFFECT", "producer arithmetic is not one wrap multiply-add")
        if len(re.findall(r"arith\.muli", task)) != 1:
            _fail("UNKNOWN_ACCESS_OR_EFFECT", "producer arithmetic is not one wrap multiply-add")
        if (product.group(2), product.group(3)) != (loads[1][0], loads[2][0]):
            _fail("ACCESS_RELATION_UNKNOWN", "producer multiply operands are not A and B")
        if (updated.group(2), updated.group(3)) != (accumulator, product.group(1)):
            _fail("STATE_CARRY_MISSING", "producer add does not carry the reduction accumulator")
        if not re.search(rf"(?m)^\s*scf\.yield\s+{re.escape(updated.group(1))}\s*:\s*i32\s*$", body):
            _fail("STATE_CARRY_MISSING", "producer reduction does not yield its updated accumulator")
        store = stores[0]
        if (store[0], store[1], tuple(x.strip() for x in store[2].split(",")),
                store[3]) != (reduction_result, block_args[3],
                               (body_indices[0], body_indices[1]), "8x8xi32"):
            _fail("ACCESS_RELATION_UNKNOWN", "producer does not write T[M,N]")
        return {
            "task_id": "producer",
            "role": "gemm_producer",
            "iteration_region": {"lower": [0, 0, 0], "upper": [8, 8, 8],
                                  "half_open": True},
            "access_relations": {"A": ["M", "K"], "B": ["K", "N"],
                                  "C_initial": ["M", "N"], "T": ["M", "N"]},
            "arithmetic": "T = C_initial + A * B",
            "state_initialization": "scf.for iter_args starts from C_initial",
            "state_carry": True,
            "reduction": {"kind": "sequential", "topology": "state_carry",
                          "k_lower": 0, "k_upper": 8, "k_step": 1},
        }

    if re.search(r"(?m)^\s*memref\.copy\b", task):
        _fail("UNKNOWN_ACCESS_OR_EFFECT", "consumer cannot initialize or copy an intermediate")
    if re.search(r"\bscf\.", task):
        _fail("REDUCTION_CONSUMER", "consumer must not contain a K reduction")
    if len(loads) != 2 or len(stores) != 1:
        _fail("REDUCTION_CONSUMER", "consumer must be one full-region elementwise add")
    expected_loads = [
        (block_args[0], (body_indices[0], body_indices[1])),
        (block_args[1], (body_indices[0], body_indices[1])),
    ]
    for actual, expected in zip(loads, expected_loads):
        actual_indices = tuple(item.strip() for item in actual[2].split(","))
        if (actual[1], actual_indices, actual[3]) != (
                expected[0], expected[1], "8x8xi32"):
            if actual[1] == expected[0] and len(actual_indices) == 2 and \
                    any(index not in body_indices for index in actual_indices):
                _fail("HALO_OR_BROADCAST",
                      "consumer accesses a region outside the producer tile")
            _fail("ACCESS_RELATION_UNKNOWN", "consumer access is not T[M,N] and E[M,N]")
    arith_ops = re.findall(r"\barith\.([A-Za-z_][A-Za-z0-9_]*)\b", task)
    if any(op not in {"constant", "addi"} for op in arith_ops) or \
            arith_ops.count("addi") != 1:
        _fail("REDUCTION_CONSUMER", "consumer contains reduction or unknown arithmetic")
    add = re.search(
        r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*arith\.addi\s+"
        r"(%[A-Za-z_][A-Za-z0-9_]*),\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*i32\s*$", task)
    if add is None or (add.group(2), add.group(3)) != (loads[0][0], loads[1][0]):
        _fail("ACCESS_RELATION_UNKNOWN", "consumer add does not compute T + E")
    store = stores[0]
    if (store[0], store[1], tuple(x.strip() for x in store[2].split(",")),
            store[3]) != (add.group(1), block_args[2],
                           (body_indices[0], body_indices[1]), "8x8xi32"):
        _fail("ACCESS_RELATION_UNKNOWN", "consumer does not write D[M,N]")
    return {
        "task_id": "consumer",
        "role": "elementwise_consumer",
        "iteration_region": {"lower": [0, 0], "upper": [8, 8],
                              "half_open": True},
        "access_relations": {"T": ["M", "N"], "E": ["M", "N"],
                              "D": ["M", "N"]},
        "arithmetic": "D = T + E",
    }


def _parse_identity(source: Any) -> dict[str, Any]:
    text, source_path = _source_text(source)
    module_matches = list(re.finditer(r"\bmodule\s*\{", text))
    if len(module_matches) != 1:
        _fail("SOURCE_STRUCTURE_INVALID", "source must contain one module")
    function_matches = list(re.finditer(
        r"func\.func\s+@([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*?)\)\s*"
        r"->\s*memref<8x8xi32>\s*\{", text, flags=re.S))
    if len(function_matches) != 1:
        _fail("SOURCE_STRUCTURE_INVALID", "source must contain one static pipeline function")
    function_match = function_matches[0]
    function = function_match.group(1)
    _, function_end = _balanced_block(text, function_match.end() - 1)
    function_body = text[function_match.end():function_end]

    args = _split_args(function_match.group(2))
    if len(args) != 5:
        _fail("ARGUMENT_CONTRACT_INVALID", "pipeline must have A, B, C_initial, E, D")
    arg_names: list[str] = []
    for index, arg in enumerate(args):
        match = re.fullmatch(
            r"(%[A-Za-z_][A-Za-z0-9_]*)\s*:\s*memref<([^>]+)>\s*"
            r"(?:\{([^}]*)\})?", arg.strip())
        if match is None:
            _fail("ARGUMENT_CONTRACT_INVALID", "pipeline argument syntax is unknown")
        name, type_text, attrs = match.groups()
        if type_text != "8x8xi32":
            _fail("DYNAMIC_SHAPE_OR_DTYPE", "all pipeline buffers must be static 8x8xi32")
        if attrs is None or attrs.strip() != "amoeba.noalias":
            _fail("ALIAS_PROOF_MISSING", f"argument {index} lacks explicit noalias")
        arg_names.append(name)
    roles = dict(zip(("A", "B", "C_initial", "E", "D"), arg_names))
    if len(set(arg_names)) != len(arg_names):
        _fail("ALIAS_PROOF_MISSING", "pipeline arguments are not distinct")

    allocs = re.findall(
        r"(?m)^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*memref\.alloc\(\)\s*:\s*"
        r"memref<([^>]+)>\s*$", function_body)
    if len(allocs) != 1 or allocs[0][1] != "8x8xi32":
        _fail("INTERMEDIATE_STORAGE_INVALID", "T must be one static internal allocation")
    t_name = allocs[0][0]
    for forbidden in ("memref.cast", "memref.subview", "memref.reinterpret_cast",
                      "memref.view", "memref.reshape", "memref.alloca"):
        if forbidden in function_body:
            _fail("ALIAS_PROOF_MISSING", f"intermediate uses unsupported aliasing op {forbidden}")

    tasks = _task_spans(function_body)
    if len(tasks) != 2:
        _fail("TASK_COUNT_INVALID", "pipeline must contain exactly producer and consumer")
    task_by_name = {match.group("name"): (match, task)
                    for match, task in tasks}
    if set(task_by_name) != {"producer", "consumer"}:
        _fail("TASK_ROLE_INVALID", "task roles must be producer and consumer")
    if len(task_by_name) != len(tasks):
        _fail("TASK_ROLE_INVALID", "task names must be unique")
    producer_match, producer = task_by_name["producer"]
    consumer_match, consumer = task_by_name["consumer"]
    producer_result = producer_match.group("result")

    producer_reads = _names(_header_segment(producer, "will_reads"))
    producer_writes = _names(_header_segment(producer, "will_writes"))
    consumer_reads = _names(_header_segment(consumer, "will_reads"))
    consumer_writes = _names(_header_segment(consumer, "will_writes"))
    if producer_reads != [roles["A"], roles["B"], roles["C_initial"]] or \
            producer_writes != [t_name]:
        _fail("TASK_CONTRACT_INVALID", "producer memory contract is not A/B/C -> T")
    if consumer_reads != [producer_result, roles["E"]] or \
            consumer_writes != [roles["D"]]:
        _fail("DATAFLOW_CONTRACT_INVALID", "consumer must consume producer state and E")

    producer_original_reads = _names(_header_segment(producer, "original_read_memrefs"))
    producer_original_writes = _names(_header_segment(producer, "original_write_memrefs"))
    consumer_original_reads = _names(_header_segment(consumer, "original_read_memrefs"))
    consumer_original_writes = _names(_header_segment(consumer, "original_write_memrefs"))
    if producer_original_reads != producer_reads or producer_original_writes != [t_name]:
        _fail("TASK_CONTRACT_INVALID", "producer original access contract is inconsistent")
    if consumer_original_reads != [t_name, roles["E"]] or \
            consumer_original_writes != [roles["D"]]:
        _fail("TASK_CONTRACT_INVALID", "consumer original access contract is inconsistent")

    # Every occurrence of the internal allocation is accounted for by the
    # declaration and the two producer/consumer original memory contracts.
    # This rejects a hidden second external user or an aliasing escape.
    t_occurrences = len(re.findall(re.escape(t_name), function_body))
    if t_occurrences != 4:
        _fail("INTERMEDIATE_EXTERNAL_USER", "T has an unaccounted-for user or alias")

    producer_graph = _validate_task_body(
        producer, "producer", producer_result, roles, t_name)
    _validate_task_body(consumer, "consumer", consumer_match.group("result"),
                        roles, t_name, producer_result)

    # The only function-level operations are the allocation, the two tasks,
    # and the return.  A second side effect or unknown control operation is a
    # semantic rejection even if the two task bodies still look canonical.
    residual = function_body
    for _, task in sorted(tasks,
                           key=lambda item: item[0].start(), reverse=True):
        match = re.search(re.escape(task), residual)
        if match:
            residual = residual[:match.start()] + "\n" + residual[match.end():]
    residual = re.sub(r"(?m)^\s*%[A-Za-z_][A-Za-z0-9_]*\s*=\s*memref\.alloc\(\)\s*:\s*"
                      r"memref<8x8xi32>\s*$", "", residual)
    residual = re.sub(r"(?m)^\s*return\s+%[A-Za-z_][A-Za-z0-9_]*\s*:\s*memref<8x8xi32>\s*$", "", residual)
    if residual.strip():
        _fail("UNKNOWN_ACCESS_OR_EFFECT", "unknown function-level operation")

    task_graph = {
        "tasks": [producer_graph, {
            "task_id": "consumer",
            "role": "elementwise_consumer",
            "iteration_region": {"lower": [0, 0], "upper": [8, 8],
                                  "half_open": True},
            "access_relations": {"T": ["M", "N"], "E": ["M", "N"],
                                  "D": ["M", "N"]},
            "arithmetic": "D = T + E",
        }],
        "edges": [{
            "source_task": "producer", "target_task": "consumer",
            "kind": "producer_consumer", "scope": "tensor_wide",
            "buffer": "T", "region": {
                "lower": [0, 0], "upper": [8, 8], "half_open": True,
            },
        }],
    }
    identity_without_id: dict[str, Any] = {
        "schema": SCHEMA,
        "source_module_sha256": _sha256(text.encode("utf-8")),
        "function": function,
        "M": 8,
        "N": 8,
        "K": 8,
        "dtype": "i32",
        "layout": "row_major_identity",
        "alias_proof": "five distinct amoeba.noalias arguments plus internal T allocation",
        "arithmetic_semantics": ARITHMETIC_SEMANTICS,
        "producer": producer_graph,
        "consumer": task_graph["tasks"][1],
        "intermediate_T": {
            "identity": "T",
            "storage": "internal_alloc",
            "shape": [8, 8],
            "dtype": "i32",
            "producer": "producer",
            "external_user_count": 0,
            "consumer": "consumer",
            "lifetime": "producer_to_consumer",
        },
        "external_inputs": ["A", "B", "C_initial", "E"],
        "external_outputs": ["D"],
        "normalized_task_graph": task_graph,
        "access_relations": {
            "producer": {"A": ["M", "K"], "B": ["K", "N"],
                          "C_initial": ["M", "N"], "T": ["M", "N"]},
            "consumer": {"T": ["M", "N"], "E": ["M", "N"],
                          "D": ["M", "N"]},
        },
    }
    identity_without_id["canonical_pipeline_id"] = _sha256(identity_without_id)
    identity_without_id["source_path"] = source_path
    return identity_without_id


def canonical_pipeline_identity(source: Any) -> dict[str, Any]:
    """Parse and return the source-bound canonical pipeline identity."""

    return _parse_identity(source)


def validate_pipeline_identity(source: Any, *, raise_on_error: bool = True) -> dict[str, Any]:
    """Validate a source and return a compact audit record.

    The default raises :class:`PipelineValidationError` on the first failed
    proof obligation.  With ``raise_on_error=False`` callers receive a stable
    ``valid``/``reason_code`` record suitable for negative fixture audits.
    """

    try:
        identity = canonical_pipeline_identity(source)
    except PipelineValidationError as error:
        if raise_on_error:
            raise
        return {"valid": False, "reason_code": error.reason_code,
                "diagnostic": str(error)}
    return {"valid": True, "reason_code": None,
            "canonical_pipeline_id": identity["canonical_pipeline_id"],
            "source_module_sha256": identity["source_module_sha256"]}


def _identity_for_reference(program: Any) -> Mapping[str, Any]:
    if isinstance(program, Mapping):
        identity = program.get("pipeline_identity", program)
        if not isinstance(identity, Mapping):
            _fail("IDENTITY_TYPE_INVALID", "pipeline_identity must be an object")
        required = ("M", "N", "K", "dtype", "arithmetic_semantics")
        if any(key not in identity for key in required):
            _fail("IDENTITY_INCOMPLETE", "pipeline identity misses numeric fields")
        return identity
    return canonical_pipeline_identity(program)


def _runtime_vector(values: Mapping[str, Any], name: str, size: int) -> list[int]:
    raw = values.get(name)
    if raw is None and name == "C_initial":
        raw = values.get("C")
    if raw is None:
        _fail("RUNTIME_INPUT_MISSING", f"runtime value {name} is missing")
    if isinstance(raw, (str, bytes)):
        _fail("RUNTIME_INPUT_INVALID", f"runtime value {name} is not a vector")
    try:
        flattened = list(raw)
    except TypeError as error:
        _fail("RUNTIME_INPUT_INVALID", f"runtime value {name} is not iterable")
        raise error
    if len(flattened) != size:
        _fail("RUNTIME_INPUT_SHAPE", f"runtime value {name} has length {len(flattened)}, expected {size}")
    if any(isinstance(item, bool) or not isinstance(item, int) or
           item < -(1 << 31) or item >= (1 << 32) for item in flattened):
        _fail("RUNTIME_INPUT_INVALID", f"runtime value {name} contains a non-i32 word")
    return [item & 0xffffffff for item in flattened]


def reference_pipeline(program: Any, values: Mapping[str, Any]) -> dict[str, list[int]]:
    """Evaluate T and D in flat row-major i32 modulo-2**32 arithmetic."""

    if not isinstance(values, Mapping):
        _fail("RUNTIME_INPUT_INVALID", "values must be a mapping")
    identity = _identity_for_reference(program)
    if identity.get("dtype") != "i32" or identity.get("arithmetic_semantics") != ARITHMETIC_SEMANTICS:
        _fail("ARITHMETIC_SEMANTICS_UNSUPPORTED", "only i32 modulo arithmetic is supported")
    m, n, k = (int(identity[key]) for key in ("M", "N", "K"))
    a = _runtime_vector(values, "A", m * k)
    b = _runtime_vector(values, "B", k * n)
    c = _runtime_vector(values, "C_initial", m * n)
    e = _runtime_vector(values, "E", m * n)
    t = list(c)
    for mi in range(m):
        for ni in range(n):
            output = mi * n + ni
            for ki in range(k):
                t[output] = (t[output] +
                             a[mi * k + ki] * b[ki * n + ni]) & 0xffffffff
    d = [(left + right) & 0xffffffff for left, right in zip(t, e)]
    return {"T": t, "D": d}


__all__ = [
    "ARITHMETIC_SEMANTICS",
    "PipelineValidationError",
    "SCHEMA",
    "canonical_pipeline_identity",
    "reference_pipeline",
    "validate_pipeline_identity",
]
