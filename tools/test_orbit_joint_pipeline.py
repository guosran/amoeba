#!/usr/bin/env python3
"""Source-bound semantic tests for the canonical producer/consumer pipeline.

The fixture is intentionally checked as a program, rather than treated as a
handwritten graph descriptor.  These tests exercise only parsing, arithmetic
semantics, and the Taskflow verifier; they do not invoke a mapper, cost model,
replay driver, or hardware tool.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from orbit_joint_pipeline import (  # noqa: E402
    ARITHMETIC_SEMANTICS,
    PipelineValidationError,
    canonical_pipeline_identity,
    reference_pipeline,
    validate_pipeline_identity,
)


SOURCE = (TOOLS.parent / "test/multi-cgra/taskflow/joint-scheduling/"
          "orbit_joint_pipeline_8x8_i32.mlir")
MLIR_OPT = Path("/tmp/amoeba-joint-scheduling-build/tools/mlir-amoeba-opt/"
                "mlir-amoeba-opt")


class CanonicalPipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.identity = canonical_pipeline_identity(SOURCE)

    def invalid(self, text: str, reason: str) -> None:
        result = validate_pipeline_identity(text, raise_on_error=False)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason_code"], reason)

    def test_identity_is_source_bound_and_proves_the_reduction(self):
        identity = self.identity
        self.assertEqual(identity["M"], 8)
        self.assertEqual(identity["N"], 8)
        self.assertEqual(identity["K"], 8)
        self.assertEqual(identity["dtype"], "i32")
        self.assertEqual(identity["arithmetic_semantics"], ARITHMETIC_SEMANTICS)
        producer = identity["producer"]
        self.assertTrue(producer["state_carry"])
        self.assertEqual(producer["state_initialization"],
                         "scf.for iter_args starts from C_initial")
        self.assertEqual(producer["reduction"]["topology"], "state_carry")
        self.assertEqual(identity["intermediate_T"]["external_user_count"], 0)
        self.assertEqual(identity["external_inputs"], ["A", "B", "C_initial", "E"])
        self.assertEqual(identity["external_outputs"], ["D"])
        self.assertIn("scf.for", self.text)
        self.assertNotIn("memref.copy", self.text)

    def test_taskflow_fixture_verifies(self):
        if not MLIR_OPT.is_file():
            self.skipTest(f"MLIR verifier is unavailable: {MLIR_OPT}")
        completed = subprocess.run(
            [str(MLIR_OPT), "--verify-each", str(SOURCE), "-o", "/dev/null"],
            text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0,
                         msg=completed.stdout + completed.stderr)

    def test_reference_uses_nonzero_c_and_e(self):
        values = {
            "A": [1 + index for index in range(64)],
            "B": [2 + index for index in range(64)],
            "C_initial": [100 + index for index in range(64)],
            "E": [200 + index for index in range(64)],
        }
        result = reference_pipeline(self.identity, values)
        expected_t = []
        expected_d = []
        for m in range(8):
            for n in range(8):
                total = values["C_initial"][m * 8 + n]
                for k in range(8):
                    total += values["A"][m * 8 + k] * values["B"][k * 8 + n]
                expected_t.append(total & 0xffffffff)
                expected_d.append((total + values["E"][m * 8 + n]) & 0xffffffff)
        self.assertEqual(result, {"T": expected_t, "D": expected_d})

    def test_reference_checks_overflow_and_signed_zero(self):
        values = {
            "A": [0xffffffff if index % 2 else 0 for index in range(64)],
            "B": [0x7fffffff if index % 3 else 0xffffffff for index in range(64)],
            "C_initial": [0xffffffff if index == 0 else 0 for index in range(64)],
            "E": [0x80000000 if index == 0 else 0 for index in range(64)],
        }
        result = reference_pipeline(self.identity, values)
        expected_t = []
        expected_d = []
        for m in range(8):
            for n in range(8):
                total = values["C_initial"][m * 8 + n]
                for k in range(8):
                    total = (total +
                             values["A"][m * 8 + k] *
                             values["B"][k * 8 + n]) & 0xffffffff
                expected_t.append(total)
                expected_d.append((total + values["E"][m * 8 + n]) & 0xffffffff)
        self.assertEqual(result["T"], expected_t)
        self.assertEqual(result["D"], expected_d)

    def test_reference_rejects_missing_or_non_i32_runtime_values(self):
        values = {name: [0] * 64 for name in ("A", "B", "C_initial", "E")}
        values.pop("E")
        with self.assertRaisesRegex(PipelineValidationError, "RUNTIME_INPUT_MISSING"):
            reference_pipeline(self.identity, values)
        values["E"] = [1] * 63 + [1 << 32]
        with self.assertRaisesRegex(PipelineValidationError, "RUNTIME_INPUT_INVALID"):
            reference_pipeline(self.identity, values)

    def test_rejects_external_intermediate_user(self):
        text = self.text.replace(
            "    return %consumed : memref<8x8xi32>",
            '    %leak = "test.external"(%T) : '
            '(memref<8x8xi32>) -> memref<8x8xi32>\n'
            "    return %consumed : memref<8x8xi32>")
        self.invalid(text, "INTERMEDIATE_EXTERNAL_USER")

    def test_rejects_missing_noalias_proof(self):
        text = self.text.replace(
            "%E: memref<8x8xi32> {amoeba.noalias}",
            "%E: memref<8x8xi32>")
        self.invalid(text, "ALIAS_PROOF_MISSING")

    def test_rejects_dynamic_control(self):
        self.invalid(self.text.replace(
            "%c8 = arith.constant 8 : index",
            "%c8 = arith.constant 7 : index"), "DYNAMIC_CONTROL")

    def test_rejects_missing_static_k_reduction(self):
        self.invalid(self.text.replace("scf.for", "taskflow.unstructured_loop", 1),
                     "UNKNOWN_ACCESS_OR_EFFECT")

    def test_rejects_reloading_c_inside_k_reduction(self):
        text = self.text.replace(
            "%c = memref.load %readC[%mi, %ni] : memref<8x8xi32>",
            "%c = memref.load %readT[%mi, %ni] : memref<8x8xi32>")
        self.invalid(text, "INITIAL_STATE_NOT_ESTABLISHED")

    def test_rejects_side_effect(self):
        text = self.text.replace(
            "        taskflow.hyperblock.yield\n      }) : (index, index) -> ()",
            "        call @side_effect() : () -> ()\n"
            "        taskflow.hyperblock.yield\n      }) : (index, index) -> ()", 1)
        self.invalid(text, "UNKNOWN_ACCESS_OR_EFFECT")

    def test_rejects_unknown_memory_access(self):
        text = self.text.replace(
            "        taskflow.hyperblock.yield\n      }) : (index, index) -> ()",
            "        memref.atomic_rmw add, %e, %writeD[%mi, %ni] : "
            "memref<8x8xi32>\n"
            "        taskflow.hyperblock.yield\n      }) : (index, index) -> ()", 1)
        self.invalid(text, "UNKNOWN_ACCESS_OR_EFFECT")

    def test_rejects_region_mismatch(self):
        old = "%n = taskflow.counter parent(%m : index) from %c0 to %c8 step %c1 : index"
        prefix, suffix = self.text.rsplit(old, 1)
        self.invalid(prefix + "%n = taskflow.counter parent(%wrong : index) from %c0 to %c8 step %c1 : index" + suffix,
                     "REGION_MISMATCH")

    def test_rejects_halo_consumer(self):
        old = "%t = memref.load %readT[%mi, %ni] : memref<8x8xi32>"
        self.invalid(self.text.replace(
            old, "%t = memref.load %readT[%mi, %ni_plus_one] : memref<8x8xi32>"),
                     "HALO_OR_BROADCAST")

    def test_rejects_reduction_consumer(self):
        text = self.text.replace(
            "        memref.store %sum, %writeD[%mi, %ni] : memref<8x8xi32>",
            "        %extra = arith.addi %e, %e : i32\n"
            "        memref.store %sum, %writeD[%mi, %ni] : memref<8x8xi32>")
        self.invalid(text, "REDUCTION_CONSUMER")


if __name__ == "__main__":
    unittest.main()
