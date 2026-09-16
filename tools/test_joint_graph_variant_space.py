#!/usr/bin/env python3
"""Transactional tests for the bounded joint graph-variant driver."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


TOOLS = Path(__file__).resolve().parent
DRIVER = TOOLS / "joint_graph_variant_space.py"


class JointGraphVariantSpaceTest(unittest.TestCase):
    def _manifest(self, path: Path) -> dict:
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]
        self.assertEqual(records[0]["record_type"], "header")
        self.assertEqual(records[-1]["record_type"], "footer")
        return {
            "header": records[0],
            "complete": records[-1]["complete"],
            "counts": {
                key: records[-1][key]
                for key in ("unique_graph_variants",
                            "duplicate_rewrite_sequences",
                            "attempted_actions")
            },
            "variants": [record for record in records
                         if record["record_type"] == "variant"],
            "duplicates": [record for record in records
                           if record["record_type"] == "duplicate"],
        }

    def _executable(self, path: Path, source: str) -> None:
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _optimizer(self, root: Path) -> Path:
        optimizer = root / "fake-opt.py"
        self._executable(
            optimizer,
            r"""
            #!/usr/bin/env python3
            import hashlib
            import json
            from pathlib import Path
            import sys

            source = Path(sys.argv[1])
            state = json.loads(source.read_text())
            history = list(state.get("history", []))
            fact_arg = next((arg for arg in sys.argv
                             if arg.startswith("--extract-joint-task-graph-facts=")),
                            None)
            if fact_arg:
                options = dict(field.split("=", 1) for field in
                               fact_arg.split("=", 1)[1].split())
                output = Path(options["output"])
                tags = sorted(history)
                legal = []
                # Every state decides its own actions.  This makes the test
                # fail if the driver carries the root action list forward.
                if not history:
                    legal = [
                        {"kind": "fusion", "mode": "producer_consumer",
                         "first_task": "producer", "second_task": "consumer"},
                        {"kind": "fusion", "mode": "sibling",
                         "first_task": "left", "second_task": "right"},
                        {"kind": "fission", "task_name": "wide",
                         "left_nodes": [0, 2]},
                        {"kind": "tiling", "task_name": "tensor", "axis": 0,
                         "factor": 2},
                    ]
                elif history == ["fission"]:
                    legal = [{"kind": "fission", "task_name": "wide.tail",
                              "left_nodes": [0]}]
                elif history == ["tile"]:
                    legal = [{"kind": "tiling", "task_name": "tensor.tile.0",
                              "axis": 1, "factor": 3}]
                structural_tags = tags
                # The two different fusion rules intentionally converge to
                # one structure, exercising canonical deduplication.
                if tags == ["sibling"]:
                    structural_tags = ["producer_consumer"]
                key_text = json.dumps(structural_tags, sort_keys=True)
                body_hash = hashlib.sha256(key_text.encode()).hexdigest()
                facts = {
                    "schema": "amoeba-joint-task-graph-facts-v1",
                    "function": "main",
                    "tasks": [{"task_name": "task", "body_sha256": body_hash}],
                    "dependencies": [],
                    "shape_needs": [{"task_name": "task", "static_dims": [12, 9]}],
                    "structural_key": {"tags": structural_tags},
                    "legal_actions": legal,
                }
                output.write_text(json.dumps(facts))
                raise SystemExit(0)

            action = next(arg for arg in sys.argv if arg.startswith(
                ("--fuse-task=", "--fission-task=", "--tile-task=")))
            if action.startswith("--fuse-task="):
                tag = "sibling" if "fusion-kind=sibling" in action \
                    else "producer_consumer"
            elif action.startswith("--fission-task="):
                tag = "fission" if "fission" not in history else "fission_again"
            else:
                tag = "tile" if "tile" not in history else "tile_again"
            state["history"] = history + [tag]
            output = Path(sys.argv[sys.argv.index("-o") + 1])
            output.write_text(json.dumps(state))
            """,
        )
        return optimizer

    def _witness(self, root: Path, fail: bool = False) -> Path:
        witness = root / "witness.py"
        self._executable(
            witness,
            """
            #!/usr/bin/env python3
            from pathlib import Path
            import sys
            source = Path(sys.argv[1]).read_text()
            candidate = Path(sys.argv[2]).read_text()
            if not source or not candidate:
                raise SystemExit(3)
            raise SystemExit({})
            """.format(7 if fail else 0),
        )
        return witness

    def _command(self, root: Path, optimizer: Path, witness: Path,
                 max_depth: int = 2) -> list[str]:
        return [
            sys.executable, str(DRIVER),
            "--input", str(root / "input.mlir"),
            "--amoeba-opt", str(optimizer),
            "--output", str(root / "manifest.json"),
            "--artifact-dir", str(root / "variants"),
            "--max-depth", str(max_depth),
            "--max-variants", "32",
            "--max-tile-factor", "4",
            "--function", "main",
            "--witness-command",
            f"{sys.executable} {witness} {{source}} {{candidate}}",
            "--timeout-seconds", "10",
        ]

    def test_discovers_all_action_kinds_recomputes_and_deduplicates(self):
        with tempfile.TemporaryDirectory(prefix="joint-graph-space-") as raw:
            root = Path(raw)
            (root / "input.mlir").write_text(json.dumps({"history": []}))
            completed = subprocess.run(
                self._command(root, self._optimizer(root), self._witness(root)),
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = self._manifest(root / "manifest.json")
            self.assertTrue(manifest["complete"])
            self.assertGreaterEqual(manifest["counts"]["unique_graph_variants"], 6)
            self.assertGreaterEqual(
                manifest["counts"]["duplicate_rewrite_sequences"], 1)
            action_kinds = {
                action["kind"]
                for variant in manifest["variants"]
                for action in variant["actions"]
            }
            self.assertEqual(action_kinds, {"fusion", "fission", "tiling"})
            fusion_modes = {
                action["mode"]
                for variant in manifest["variants"]
                for action in variant["actions"]
                if action["kind"] == "fusion"
            }
            # Sibling reaches the same structure as producer-consumer and is
            # therefore represented in the duplicate rewrite provenance.
            fusion_modes.update(
                action["mode"]
                for duplicate in manifest["duplicates"]
                for action in duplicate["actions"]
                if action["kind"] == "fusion"
            )
            self.assertEqual(fusion_modes, {"producer_consumer", "sibling"})
            self.assertTrue(any(len(variant["actions"]) == 2
                                for variant in manifest["variants"]))
            self.assertTrue(all(
                variant["semantic_witness"]["status"] == "verified"
                for variant in manifest["variants"]
            ))
            self.assertTrue(all(
                "command" not in variant["semantic_witness"]
                for variant in manifest["variants"]
            ))
            self.assertTrue(all(
                ".joint-graph-variants-" not in
                json.dumps(variant["semantic_witness"])
                for variant in manifest["variants"]
            ))
            self.assertEqual(len(list((root / "variants").glob("*.mlir"))),
                             len(manifest["variants"]))
            self.assertEqual(
                manifest["header"]["preparation"]["contract"],
                "amoeba-graph-variant-preparation-v1")
            self.assertTrue(manifest["header"]["preparation"]
                            ["data_movement"]["required_exactly_once"])
            for variant in manifest["variants"]:
                self.assertEqual(variant["graph_variant_sha256"],
                                 variant["result_ir"]["sha256"])
                self.assertEqual(variant["result_ir"]["path"],
                                 variant["result_ir"]["artifact"])
                self.assertFalse(Path(variant["result_ir"]["path"])
                                 .is_absolute())
            artifact_path = Path(os.path.normpath(
                str((root / "manifest.json").parent /
                    manifest["header"]["artifact_dir"]["path"])))
            self.assertEqual(artifact_path, root / "variants")
            bundle = root / "bundle"
            bundle.mkdir()
            (root / "manifest.json").rename(bundle / "manifest.json")
            (root / "variants").rename(bundle / "variants")
            moved = root / "moved"
            bundle.rename(moved)
            moved_manifest = self._manifest(moved / "manifest.json")
            moved_artifact_path = Path(os.path.normpath(
                str((moved / "manifest.json").parent /
                    moved_manifest["header"]["artifact_dir"]["path"])))
            for variant in moved_manifest["variants"]:
                self.assertTrue((moved_artifact_path /
                                 variant["result_ir"]["path"]).is_file())

    def test_witness_failure_rolls_back_manifest_and_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="joint-graph-rollback-") as raw:
            root = Path(raw)
            (root / "input.mlir").write_text(json.dumps({"history": []}))
            completed = subprocess.run(
                self._command(root, self._optimizer(root),
                              self._witness(root, fail=True), max_depth=1),
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("semantic witness failed closed", completed.stderr)
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse((root / "variants").exists())

    def test_rejects_unverified_run_and_budget_before_publication(self):
        with tempfile.TemporaryDirectory(prefix="joint-graph-contract-") as raw:
            root = Path(raw)
            (root / "input.mlir").write_text(json.dumps({"history": []}))
            optimizer = self._optimizer(root)
            command = self._command(root, optimizer, self._witness(root), 1)
            witness_index = command.index("--witness-command")
            del command[witness_index:witness_index + 2]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((root / "manifest.json").exists())

            command = self._command(root, optimizer, self._witness(root), 1)
            command[command.index("--max-variants") + 1] = "1"
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=False)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("budget exceeded", completed.stderr)
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse((root / "variants").exists())


if __name__ == "__main__":
    raise SystemExit(unittest.main())
