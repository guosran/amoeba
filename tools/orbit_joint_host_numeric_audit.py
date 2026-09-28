#!/usr/bin/env python3
"""Execute materialized C++ Taskflow witnesses on the host and check every write.

Python supplies inputs and expected i32 values only.  The computation under
test is the generated Taskflow IR, lowered by the C++ host SCF pass and JIT run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from orbit_joint_pass_regression import _pass_flag
from orbit_joint_reference_audit import _cases


def _run(command: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(command, input=input_text, text=True,
                            capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr[-4000:]}")
    return result.stdout


def _wrap(value: int) -> int:
    return ((value + 2**31) % 2**32) - 2**31


def _dense(values: list[int]) -> str:
    rows = [", ".join(str(_wrap(x)) for x in values[r:r + 8])
            for r in range(0, 64, 8)]
    return "dense<[" + ", ".join("[" + row + "]" for row in rows) + "]>"


def _region(attributes: str) -> tuple[int, int, int, int]:
    lower = re.search(r'amoeba.tiling.output_region_lowers = \[array<i64: (\d+), (\d+)>\]', attributes)
    upper = re.search(r'amoeba.tiling.output_region_uppers = \[array<i64: (\d+), (\d+)>\]', attributes)
    if not lower or not upper:
        raise ValueError("missing output tile region on task completion")
    return int(lower[1]), int(upper[1]), int(lower[2]), int(upper[2])


def _traces(host: str) -> list[dict]:
    records: list[dict] = []
    pending: dict | None = None
    for line in host.splitlines():
        if "amoeba.host.task_completion" in line:
            kind = re.search(r'amoeba.semantic.task_kind = "([^"]+)"', line)
            name = re.search(r'amoeba.host.task_name = "([^"]+)"', line)
            k = re.search(r'amoeba.semantic.K_range = array<i64: (\d+), (\d+)>', line)
            if not k:
                k = re.search(r'amoeba.semantic.k_range = array<i64: (\d+), (\d+)>', line)
            if not kind or not name:
                raise ValueError("missing kind/name on task completion")
            pending = {"kind": kind[1], "name": name[1], "k_range":
                       [int(k[1]), int(k[2])] if k else [0, 8],
                       "region": _region(line),
                       "policy": "parallel" if 'amoeba.semantic.k_policy = "parallel"' in line else "sequential",
                       "reduction_final": "amoeba.semantic.reduction_final" in line,
                       "copies": 0}
        if "amoeba.host.trace_symbol" in line:
            if pending is None:
                raise ValueError("trace copy without task completion")
            symbol = re.search(r'amoeba.host.trace_symbol = "([^"]+)"', line)
            write = re.search(r'amoeba.host.trace_write_index = (\d+) : i64', line)
            ordinal = re.search(r'amoeba.host.trace_task_index = (\d+) : i64', line)
            if not symbol or not write or not ordinal:
                raise ValueError("incomplete trace copy attributes")
            records.append({**pending, "symbol": symbol[1],
                            "write_index": int(write[1]),
                            "task_index": int(ordinal[1])})
            pending["copies"] += 1
    if not records or pending is None or pending["copies"] == 0:
        raise ValueError("host lowering did not publish all task traces")
    return records


def _expected_trace(trace: dict, values: dict[str, list[int]]) -> list[int]:
    kind = trace["kind"]
    k0, k1 = trace["k_range"]
    expected = [0] * 64
    for m in range(8):
        for n in range(8):
            offset = m * 8 + n
            product = sum(values["A"][m * 8 + k] * values["B"][k * 8 + n]
                          for k in range(k0, k1))
            if kind == "gemm_producer":
                if trace["policy"] == "sequential":
                    # Sequential blocks accumulate the prefix, including C once.
                    product = sum(values["A"][m * 8 + k] * values["B"][k * 8 + n]
                                  for k in range(k1))
                    expected[offset] = values["C_initial"][offset] + product
                else:
                    expected[offset] = product
            elif kind == "reduction":
                expected[offset] = product
                if trace["reduction_final"]:
                    expected[offset] += values["C_initial"][offset]
            elif kind in ("elementwise_consumer", "fused_producer_consumer",
                          "fused_reduction_consumer"):
                expected[offset] = (values["C_initial"][offset] +
                                    sum(values["A"][m * 8 + k] *
                                        values["B"][k * 8 + n] for k in range(8)) +
                                    values["E"][offset])
            else:
                raise ValueError(f"unsupported trace kind {kind}")
    return expected


def _category(trace: dict) -> str:
    kind = trace["kind"]
    if kind == "gemm_producer":
        return ("state_carry" if trace["policy"] == "sequential"
                else "private_partial")
    return "reduction_output" if kind == "reduction" else "consumer_output"


def _runner_module(host: str, traces: list[dict], *,
                   negative_control: bool = False) -> tuple[str, dict]:
    if not host.rstrip().endswith("}"):
        raise ValueError("expected one MLIR module")
    prefix = host.rstrip()[:-1]
    lines = [prefix]
    lines += ["  func.func @__orbit_compare(%actual: memref<8x8xi32>, %expected: memref<8x8xi32>, %m0: index, %m1: index, %n0: index, %n1: index) -> i32 {",
              "    %zero = arith.constant 0 : i32",
              "    %one = arith.constant 1 : index",
              "    %count = scf.for %m = %m0 to %m1 step %one iter_args(%sum = %zero) -> i32 {",
              "      %inner = scf.for %n = %n0 to %n1 step %one iter_args(%local = %sum) -> i32 {",
              "        %a = memref.load %actual[%m, %n] : memref<8x8xi32>",
              "        %e = memref.load %expected[%m, %n] : memref<8x8xi32>",
              "        %bad = arith.cmpi ne, %a, %e : i32",
              "        %delta = arith.extui %bad : i1 to i32",
              "        %next = arith.addi %local, %delta : i32",
              "        scf.yield %next : i32",
              "      }",
              "      scf.yield %inner : i32",
              "    }",
              "    return %count : i32",
              "  }"]
    cases = _cases()
    globals_: list[str] = []
    body: list[str] = ["  func.func @main() -> i32 {",
                       "    %c0 = arith.constant 0 : i32",
                       "    %i0 = arith.constant 0 : index",
                       "    %i8 = arith.constant 8 : index"]
    current = "%c0"
    checks = {"final_D": 0, "state_carry": 0, "private_partial": 0,
              "reduction_output": 0, "consumer_output": 0}
    for case_index, (_, values) in enumerate(cases):
        reference_d = _expected_trace({"kind": "elementwise_consumer",
                                       "k_range": [0, 8]}, values)
        if negative_control and case_index == 0:
            reference_d[0] += 1
        for name in ("A", "B", "C_initial", "E"):
            symbol = f"__orbit_input_{case_index}_{name}"
            globals_.append(f"  memref.global constant @{symbol} : memref<8x8xi32> = {_dense(values[name])}")
            body.append(f"    %in_{case_index}_{name} = memref.get_global @{symbol} : memref<8x8xi32>")
        d_symbol = f"__orbit_output_{case_index}"
        globals_.append(f"  memref.global @{d_symbol} : memref<8x8xi32> = uninitialized")
        body.append(f"    %out_{case_index} = memref.get_global @{d_symbol} : memref<8x8xi32>")
        body.append(f"    %ret_{case_index} = func.call @gemm_then_add_8x8_i32(%in_{case_index}_A, %in_{case_index}_B, %in_{case_index}_C_initial, %in_{case_index}_E, %out_{case_index}) : (memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>) -> memref<8x8xi32>")
        for trace_index, trace in enumerate(traces):
            symbol = f"__orbit_expected_{case_index}_{trace_index}"
            globals_.append(f"  memref.global constant @{symbol} : memref<8x8xi32> = {_dense(_expected_trace(trace, values))}")
            base = f"t{case_index}_{trace_index}"
            body.append(f"    %actual_{base} = memref.get_global @{trace['symbol']} : memref<8x8xi32>")
            body.append(f"    %expected_{base} = memref.get_global @{symbol} : memref<8x8xi32>")
            m0, m1, n0, n1 = trace["region"]
            for label, value in (("m0", m0), ("m1", m1), ("n0", n0), ("n1", n1)):
                body.append(f"    %{label}_{base} = arith.constant {value} : index")
            body.append(f"    %bad_{base} = func.call @__orbit_compare(%actual_{base}, %expected_{base}, %m0_{base}, %m1_{base}, %n0_{base}, %n1_{base}) : (memref<8x8xi32>, memref<8x8xi32>, index, index, index, index) -> i32")
            new = f"%sum_{base}"
            body.append(f"    {new} = arith.addi {current}, %bad_{base} : i32")
            current = new
            checks[_category(trace)] += 1
        symbol = f"__orbit_final_D_{case_index}"
        globals_.append(f"  memref.global constant @{symbol} : memref<8x8xi32> = {_dense(reference_d)}")
        body.append(f"    %expected_D_{case_index} = memref.get_global @{symbol} : memref<8x8xi32>")
        body.append(f"    %bad_D_{case_index} = func.call @__orbit_compare(%out_{case_index}, %expected_D_{case_index}, %i0, %i8, %i0, %i8) : (memref<8x8xi32>, memref<8x8xi32>, index, index, index, index) -> i32")
        new = f"%sum_D_{case_index}"
        body.append(f"    {new} = arith.addi {current}, %bad_D_{case_index} : i32")
        current = new
        checks["final_D"] += 1
    body += [f"    return {current} : i32", "  }", "}"]
    return "\n".join(lines + globals_ + body) + "\n", checks


def run(source: Path, manifest_path: Path, optimizer: Path, mlir_opt: Path,
        runner: Path, runner_utils: Path, limit: int | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    candidates = manifest["candidates"][:limit]
    records = []
    negative_control_mismatches = None
    with tempfile.TemporaryDirectory() as tmp:
        for index, candidate in enumerate(candidates):
            if not candidate["witness"]["executable"]:
                raise ValueError(f"candidate {index} lacks executable witness")
            pipeline = [_pass_flag(step) for step in candidate["witness"]["pass_pipeline"]]
            host = _run([str(optimizer), str(source), "--verify-each", *pipeline,
                         "--lower-joint-taskflow-to-host-scf=capture-task-states"])
            traces = _traces(host)
            module, checks = _runner_module(host, traces)
            path = Path(tmp) / f"candidate_{index}.mlir"
            path.write_text(module)
            llvm_ir = _run([str(mlir_opt), str(path), "--convert-scf-to-cf",
                            "--convert-cf-to-llvm", "--convert-arith-to-llvm",
                            "--finalize-memref-to-llvm", "--convert-func-to-llvm",
                            "--reconcile-unrealized-casts"])
            llvm_path = Path(tmp) / f"candidate_{index}_llvm.mlir"
            llvm_path.write_text(llvm_ir)
            output = _run([str(runner), str(llvm_path), "-e", "main",
                           "-entry-point-result=i32", "-shared-libs",
                           str(runner_utils)])
            mismatch = int(output.strip().splitlines()[-1])
            element_checks = {key: 0 for key in checks}
            element_checks["final_D"] = 64
            for trace in traces:
                m0, m1, n0, n1 = trace["region"]
                element_checks[_category(trace)] += (m1 - m0) * (n1 - n0)
            records.append({"graph_id": candidate["graph_id"],
                            "task_writes_observed": len(traces),
                            "checks_per_case": {k: v // 4 for k, v in checks.items()},
                            "element_checks_per_case": element_checks,
                            "element_mismatches": mismatch, "pass": mismatch == 0})
            if index == 0:
                negative_module, _ = _runner_module(host, traces,
                                                    negative_control=True)
                negative_path = Path(tmp) / "negative_control.mlir"
                negative_path.write_text(negative_module)
                negative_llvm = _run([str(mlir_opt), str(negative_path),
                                      "--convert-scf-to-cf", "--convert-cf-to-llvm",
                                      "--convert-arith-to-llvm",
                                      "--finalize-memref-to-llvm",
                                      "--convert-func-to-llvm",
                                      "--reconcile-unrealized-casts"])
                negative_llvm_path = Path(tmp) / "negative_control_llvm.mlir"
                negative_llvm_path.write_text(negative_llvm)
                negative_output = _run([str(runner), str(negative_llvm_path),
                                        "-e", "main", "-entry-point-result=i32",
                                        "-shared-libs", str(runner_utils)])
                negative_control_mismatches = int(
                    negative_output.strip().splitlines()[-1])
                if negative_control_mismatches != 1:
                    raise ValueError("host elementwise negative control did "
                                     "not detect exactly one mismatch")
    return {"schema": "orbit-joint-generated-taskflow-host-numeric-audit-v1",
            "execution": "C++ Taskflow witness -> C++ host SCF lowering -> LLVM JIT",
            "candidate_count": len(records), "case_count": 4,
            "negative_control_mismatches": negative_control_mismatches,
            "candidate_case_comparisons": 4 * len(records),
            "all_pass": all(row["pass"] for row in records),
            "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "amoeba-opt", "mlir-opt", "runner",
                 "runner-utils", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run(args.source, args.manifest, args.amoeba_opt, args.mlir_opt,
                 args.runner, args.runner_utils, args.limit)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"},
                     sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
