#!/usr/bin/env python3
"""Verify every executable C++ rewrite witness against exact Taskflow facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from orbit_joint_facts_compare import (GraphFactsMismatch,
                                       compare_graph_facts_by_semantic_signature)
from orbit_joint_pass_regression import _pass_flag


def run(source: Path, manifest_path: Path, optimizer: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("C++ rewrite manifest is incomplete")
    records = []
    with tempfile.TemporaryDirectory() as directory:
        for index, candidate in enumerate(manifest["candidates"]):
            witness = candidate["witness"]
            if not witness["executable"]:
                continue
            facts_path = Path(directory) / f"{index}.json"
            command = [str(optimizer), str(source), "--verify-each"]
            command.extend(_pass_flag(step) for step in
                           witness["pass_pipeline"])
            command.extend((f"--extract-joint-task-graph-facts=output={facts_path}",
                            "-o", "/dev/null"))
            process = subprocess.run(command, text=True, capture_output=True)
            facts_match = False
            reason = process.stderr[-1000:] if process.returncode else ""
            if process.returncode == 0 and facts_path.is_file():
                try:
                    comparison = compare_graph_facts_by_semantic_signature(
                        candidate,
                        json.loads(facts_path.read_text(encoding="utf-8")))
                    facts_match = comparison["pass"]
                    if not facts_match:
                        reason = "typed graph facts differ"
                except (GraphFactsMismatch, ValueError) as error:
                    reason = str(error)
            records.append({"graph_id": candidate["graph_id"],
                            "pipeline": witness["pass_pipeline"],
                            "verifier_pass": process.returncode == 0,
                            "facts_match": facts_match,
                            "reason": reason})
    return {"schema": "orbit-joint-all-executable-witness-replay-v1",
            "owner": "C++ standalone MLIR passes",
            "candidate_count": len(manifest["candidates"]),
            "executable_witness_count": len(records),
            "verified_count": sum(row["verifier_pass"] for row in records),
            "exact_graph_facts_match_count": sum(row["facts_match"] for row in records),
            "all_pass": all(row["verifier_pass"] and row["facts_match"]
                        for row in records),
            "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.manifest, args.amoeba_opt)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "records"}, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
