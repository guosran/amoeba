#!/usr/bin/env python3
"""Focused semantic tests for the hardware-independent GEMM task space."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from orbit_task_variant_space import (  # noqa: E402
    VariantError,
    VariantValidationError,
    canonical_program_identity,
    enumeration_statistics,
    enumerate_variants,
    interpret_variant,
    recognize,
    reference,
    validate_variant,
)


CANONICAL = Path(
    "/tmp/amoeba-joint-scheduling-review/orbit_vectorcgra_memory_blocking_review/"
    "orbit_generic_blocking_mainline/gemm_8x8_i32.mlir"
)


def _descriptor(m: int = 8, n: int = 8, k: int = 8,
                dtype: str = "i32") -> dict:
    operation = {
        "access": {
            "A": ["M", "K"],
            "B": ["K", "N"],
            "C_read": ["M", "N"],
            "C_write": ["M", "N"],
        },
        "arithmetic": "i32.wrap(C+A*B)" if dtype == "i32" else "f32",
        "loop_order": ["M", "N", "K"],
        "alias": "three distinct amoeba.noalias args",
    }
    canonical = json.dumps(operation, sort_keys=True, separators=(",", ":"))
    payload = {
        "function": "fixture_gemm",
        "M": m,
        "N": n,
        "K": k,
        "dtype": dtype,
        "layout": "row_major_identity",
        "C_initialization": "caller supplied, read before each first accumulation",
        "alias_proof": "three distinct amoeba.noalias args",
        "operation": operation,
    }
    source_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **payload,
        "source_program_sha256": source_hash,
        "operation_access_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _values(m: int = 8, n: int = 8, k: int = 8) -> dict[str, list[int]]:
    return {
        "A": [((index * 13 + 5) % 31) - 15 for index in range(m * k)],
        "B": [((index * 7 + 3) % 23) - 11 for index in range(k * n)],
        "C_initial": [((index * 5 + 9) % 17) - 8 for index in range(m * n)],
    }


class OrbitTaskVariantSpaceTest(unittest.TestCase):
    def test_canonical_source_identity_matches_existing_recognizer_fields(self):
        descriptor = recognize(CANONICAL)
        self.assertEqual(
            (descriptor["M"], descriptor["N"], descriptor["K"], descriptor["dtype"]),
            (8, 8, 8, "i32"),
        )
        self.assertEqual(descriptor["layout"], "row_major_identity")
        self.assertIn("noalias", descriptor["alias_proof"])
        self.assertEqual(len(descriptor["canonical_program_id"]), 64)

    def test_cartesian_enumeration_is_deterministic_and_backend_neutral(self):
        program = _descriptor()
        left = enumerate_variants(
            program, [8, 1, 4, 2, 4], [4, 8, 1, 2], [8, 2, 4, 1],
            {"parallel_partial_reduce", "sequential_accumulate"},
        )
        right = enumerate_variants(
            program, [2, 1, 8, 4], [2, 1, 8, 4], [4, 1, 8, 2],
            ["sequential_accumulate", "parallel_partial_reduce"],
        )
        self.assertEqual(
            [variant["variant_id"] for variant in left],
            [variant["variant_id"] for variant in right],
        )
        self.assertEqual(len(left), 4 * 4 * 4 * 3)
        self.assertTrue(all(variant["semantic_legal"] for variant in left))
        self.assertTrue(all(variant["backend_status"] == "unknown"
                            for variant in left))
        self.assertEqual(
            {(variant["k_policy"], variant["reduction_topology"])
             for variant in left},
            {("sequential_accumulate", "none"),
             ("parallel_partial_reduce", "linear"),
             ("parallel_partial_reduce", "balanced_tree")},
        )

    def test_reject_and_exact_tail_legalization(self):
        program = _descriptor(5, 7, 6)
        self.assertEqual(
            enumerate_variants(program, [4], [4], [4],
                                ["sequential_accumulate"], "reject"),
            [],
        )
        variants = enumerate_variants(
            program, [4], [4], [4], ["sequential_accumulate"], "exact"
        )
        self.assertEqual(len(variants), 1)
        variant = variants[0]
        self.assertTrue(variant["tail_required"])
        self.assertTrue(variant["semantic_tail_supported"])
        self.assertEqual(variant["backend_tail_supported"], "unknown")
        self.assertEqual(
            [(tile["m_range"], tile["n_range"]) for tile in variant["output_tiles"]],
            [([0, 4], [0, 4]), ([0, 4], [4, 7]),
             ([4, 5], [0, 4]), ([4, 5], [4, 7])],
        )
        self.assertEqual(
            [block["k_range"] for block in variant["k_blocks"]],
            [[0, 4], [4, 6]],
        )
        self.assertTrue(validate_variant(program, variant))
        oversized = enumerate_variants(
            program, [5, 9], [7, 10], [6, 11],
            ["sequential_accumulate"], "exact")
        self.assertEqual(len(oversized), 1)
        self.assertEqual((oversized[0]["BM"], oversized[0]["BN"],
                          oversized[0]["BK"]), (5, 7, 6))

    def test_deterministic_rejection_histogram(self):
        tail = enumeration_statistics(
            _descriptor(5, 7, 6), [4], [4], [4],
            ["sequential_accumulate", "parallel_partial_reduce"],
            "reject")
        self.assertEqual(tail["requested_combination_count"], 3)
        self.assertEqual(tail["rejection_reason_histogram"],
                         {"nondivisible_static_tail": 3})
        f32 = enumeration_statistics(
            _descriptor(dtype="f32"), [4], [4], [4],
            ["sequential_accumulate", "parallel_partial_reduce"])
        self.assertEqual(f32["legal_raw_combination_count"], 1)
        self.assertEqual(f32["rejection_reason_histogram"],
                         {"f32_reassociation_not_permitted": 2})

    def test_sequential_graph_has_state_carry_and_completion_only_join(self):
        program = _descriptor()
        variant = enumerate_variants(
            program, [4], [4], [4], ["sequential_accumulate"]
        )[0]
        validate_variant(program, variant)
        self.assertEqual(
            {edge["kind"] for edge in variant["edges"]},
            {"state_carry", "completion_only"},
        )
        self.assertEqual(
            sum(edge["kind"] == "state_carry" for edge in variant["edges"]),
            4,
        )
        self.assertFalse(any(edge["kind"] == "reduction_input"
                             for edge in variant["edges"]))
        compute = [task for task in variant["tasks"]
                   if task["task_kind"] == "gemm"]
        self.assertEqual(sum(task["initial_C_ownership"] == "external_initial_C"
                             for task in compute), 4)
        self.assertEqual(sum(task["final_C_ownership"] == "external_final_C"
                             for task in compute), 4)

    def test_parallel_graph_uses_private_partials_and_explicit_reduction(self):
        program = _descriptor()
        variant = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"]
        )[0]
        validate_variant(program, variant)
        compute = [task for task in variant["tasks"]
                   if task["task_kind"] == "gemm"]
        reductions = [task for task in variant["tasks"]
                      if task["task_kind"] == "reduction"]
        self.assertEqual(len(compute), 8)
        self.assertEqual(len(reductions), 8)
        self.assertTrue(all(task["C_output_region"] is None for task in compute))
        partial_buffers = [task["partial_C_region"]["buffer"] for task in compute]
        self.assertEqual(len(partial_buffers), len(set(partial_buffers)))
        self.assertEqual(sum(task["C_output_region"]["buffer"] == "C"
                             for task in reductions), 4)
        self.assertEqual(
            sum(edge["kind"] == "reduction_input" for edge in variant["edges"]),
            8,
        )

    def test_linear_and_balanced_topologies_are_distinct_and_cover_one_block(self):
        program = _descriptor()
        linear = enumerate_variants(
            program, [4], [4], [8], ["parallel_partial_reduce"],
            reduction_topology="linear",
        )[0]
        tree = enumerate_variants(
            program, [4], [4], [8], ["parallel_partial_reduce"],
            reduction_topology="balanced_tree",
        )[0]
        self.assertNotEqual(linear["variant_id"], tree["variant_id"])
        for variant in (linear, tree):
            self.assertTrue(validate_variant(program, variant))
            reductions = [task for task in variant["tasks"]
                          if task["task_kind"] == "reduction"]
            self.assertEqual(len(reductions), 4)
            self.assertTrue(all(task["final_C_ownership"] == "external_final_C"
                                for task in reductions))
            compute = [task for task in variant["tasks"]
                       if task["task_kind"] == "gemm"]
            self.assertTrue(all(task["partial_lifetime"]["last_consumer"]
                                for task in compute))
            self.assertEqual(
                sum(task["C_output_region"]["buffer"] == "C"
                    for task in reductions), 4)

    def test_topology_mutation_missing_partial_and_concurrent_final_write_fail(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"],
            reduction_topology="balanced_tree",
        )[0]
        mutations = (
            lambda item: item.__setitem__("reduction_topology", "linear"),
            lambda item: item["tasks"][0].__setitem__("partial_C_region", None),
            lambda item: item["tasks"][0].__setitem__("C_output_region", {
                "schema": "orbit-half-open-region-v1", "buffer": "C",
                "lower": [0, 0], "upper": [4, 4], "half_open": True,
            }),
        )
        for mutate in mutations:
            candidate = deepcopy(original)
            mutate(candidate)
            with self.assertRaises(VariantValidationError):
                validate_variant(program, candidate)

    def test_validator_rejects_missing_predecessor_and_duplicate_initial_c(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"],
            reduction_topology="balanced_tree",
        )[0]
        candidate = deepcopy(original)
        candidate["tasks"][-1]["predecessors"].pop()
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

        original = enumerate_variants(
            program, [4], [4], [2], ["parallel_partial_reduce"],
            reduction_topology="balanced_tree",
        )[0]
        candidate = deepcopy(original)
        reductions = [task for task in candidate["tasks"]
                      if task["task_kind"] == "reduction" and
                      task["logical_output_tile_id"] == "tile_m0_n0"]
        self.assertGreater(len(reductions), 1)
        reductions[1]["initial_C_ownership"] = "reduction_task_once"
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

    def test_validator_rejects_missing_final_c_and_join_as_reduction(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"],
            reduction_topology="linear",
        )[0]
        candidate = deepcopy(original)
        final = [task for task in candidate["tasks"]
                 if task.get("final_C_ownership") == "external_final_C"]
        self.assertEqual(len(final), 4)
        final[0]["C_output_region"] = None
        final[0]["regions"]["C_output"] = None
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

        candidate = deepcopy(original)
        join = next(task for task in candidate["tasks"]
                    if task["task_kind"] == "completion_join")
        join["task_kind"] = "reduction"
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

    def test_validator_rejects_k_gap_and_overlap(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["sequential_accumulate"]
        )[0]
        gemm = [task for task in original["tasks"]
                if task["task_kind"] == "gemm"]
        self.assertEqual(len(gemm), 8)
        for mutation in (lambda task: task["k_range"].__setitem__(0, 5),
                         lambda task: task["k_range"].__setitem__(1, 3)):
            candidate = deepcopy(original)
            mutation(next(task for task in candidate["tasks"]
                          if task["task_kind"] == "gemm" and
                          task["k_range"] == [0, 4]))
            with self.assertRaises(VariantValidationError):
                validate_variant(program, candidate)

    def test_validator_rejects_missing_or_duplicate_partial(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"],
            reduction_topology="linear",
        )[0]
        partials = [task for task in original["tasks"]
                    if task["task_kind"] == "gemm"]
        self.assertEqual(len(partials), 8)
        candidate = deepcopy(original)
        candidate_partials = [task for task in candidate["tasks"]
                              if task["task_kind"] == "gemm"]
        candidate_partials[0]["partial_C_region"] = None
        candidate_partials[0]["regions"]["partial_C"] = None
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

        candidate = deepcopy(original)
        candidate_partials = [task for task in candidate["tasks"]
                              if task["task_kind"] == "gemm"]
        duplicate = deepcopy(candidate_partials[1]["partial_C_region"])
        candidate_partials[0]["partial_C_region"] = duplicate
        candidate_partials[0]["regions"]["partial_C"] = duplicate
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

    def test_validator_rejects_output_gap_and_overlap(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["sequential_accumulate"]
        )[0]
        self.assertEqual(len(original["output_tiles"]), 4)
        candidate = deepcopy(original)
        candidate["output_tiles"].pop()
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

        candidate = deepcopy(original)
        candidate["output_tiles"][1] = deepcopy(candidate["output_tiles"][0])
        with self.assertRaises(VariantValidationError):
            validate_variant(program, candidate)

    def test_validator_rejects_mutated_regions_edges_and_identity(self):
        program = _descriptor()
        original = enumerate_variants(
            program, [4], [4], [4], ["parallel_partial_reduce"]
        )[0]
        for mutate in (
            lambda item: item["tasks"][0]["A_read_region"]["upper"].__setitem__(1, 99),
            lambda item: item["tasks"][0].__setitem__("C_output_region", {
                "schema": "orbit-half-open-region-v1", "buffer": "C",
                "lower": [0, 0], "upper": [4, 4], "half_open": True,
            }),
            lambda item: item["edges"][0].__setitem__("kind", "state_carry_bad"),
            lambda item: item.__setitem__("variant_id", "0" * 64),
        ):
            candidate = deepcopy(original)
            mutate(candidate)
            with self.assertRaises(VariantValidationError):
                validate_variant(program, candidate)

    def test_interpreters_match_reference_for_signed_inputs_and_nonzero_c(self):
        program = _descriptor()
        values = _values()
        expected = reference(program, values)
        for policy in ("sequential_accumulate", "parallel_partial_reduce"):
            variant = enumerate_variants(program, [1, 2, 4], [2, 4], [2, 4],
                                         [policy])[3]
            self.assertEqual(interpret_variant(program, variant, values), expected)
        # The legacy scalar-shape spelling remains deterministic and returns
        # the same unsigned i32 words.
        self.assertEqual(reference(8, 8, 8, values), expected)

    def test_interpreters_match_for_multiple_shapes_and_exact_tails(self):
        for shape in ((3, 5, 4), (5, 7, 6), (8, 8, 8)):
            m, n, k = shape
            program = _descriptor(m, n, k)
            values = _values(m, n, k)
            expected = reference(program, values)
            for policy, topology in (
                    ("sequential_accumulate", "none"),
                    ("parallel_partial_reduce", "linear"),
                    ("parallel_partial_reduce", "balanced_tree")):
                variants = enumerate_variants(
                    program, [2, 4], [2, 4], [2, 4], [policy],
                    tail_policy="exact", reduction_topology=topology,
                )
                self.assertTrue(variants)
                for variant in variants:
                    self.assertEqual(interpret_variant(program, variant, values),
                                     expected)

    def test_semantic_f32_variant_is_legal_but_i32_interpreter_is_fail_closed(self):
        program = _descriptor(dtype="f32")
        variants = enumerate_variants(
            program, [4], [4], [4],
            ["sequential_accumulate", "parallel_partial_reduce"])
        self.assertEqual(
            {(variant["k_policy"], variant["reduction_topology"])
             for variant in variants},
            {("sequential_accumulate", "none")},
        )
        self.assertTrue(all(variant["semantic_legal"] for variant in variants))
        for variant in variants:
            validate_variant(program, variant)
        with self.assertRaises(VariantError):
            interpret_variant(program, variants[0], _values())

    def test_variant_ids_are_independent_of_python_hash_seed(self):
        program = _descriptor()
        source = (
            "import json, sys; "
            "sys.path.insert(0, %r); "
            "from orbit_task_variant_space import enumerate_variants; "
            "p = %s; "
            "v = enumerate_variants(p, {8, 1, 4, 2}, {4, 8, 1, 2}, "
            "{8, 2, 4, 1}, {'parallel_partial_reduce', "
            "'sequential_accumulate'}); "
            "print(json.dumps([x['variant_id'] for x in v]))"
        ) % (str(TOOLS), repr(program))
        outputs = []
        for seed in ("1", "777"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(subprocess.check_output(
                [sys.executable, "-c", source],
                cwd=str(TOOLS.parent), env=environment, text=True,
            ).strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_source_parser_rejects_unknown_side_effect(self):
        text = CANONICAL.read_text(encoding="utf-8")
        mutated = text.replace(
            "          %product = arith.muli %a, %b : i32\n",
            "          call @unknown() : () -> ()\n"
            "          %product = arith.muli %a, %b : i32\n",
        )
        path = Path(self.id()).with_suffix(".mlir")
        # A deterministic temporary location is unnecessary here; source
        # parser failures happen before the path is used for any artifact.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".mlir") as stream:
            stream.write(mutated)
            stream.flush()
            with self.assertRaises(VariantError):
                recognize(Path(stream.name))

    def test_source_parser_rejects_ssa_redefinition_and_wrong_step(self):
        text = CANONICAL.read_text(encoding="utf-8")
        mutations = (
            text.replace(
                "%a = memref.load %A[%m, %k]",
                "%m = memref.load %A[%m, %k]",
            ),
            text.replace(
                "%c1 = arith.constant 1 : index",
                "%c1 = arith.constant 2 : index",
            ),
        )
        import tempfile
        for mutated in mutations:
            with tempfile.NamedTemporaryFile("w", suffix=".mlir") as stream:
                stream.write(mutated)
                stream.flush()
                with self.assertRaises(VariantError):
                    recognize(Path(stream.name))


if __name__ == "__main__":
    unittest.main()
