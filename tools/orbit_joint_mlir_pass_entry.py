#!/usr/bin/env python3
"""Opt-in ORBIT entry for the MLIR-owned joint semantic rewrite pass.

The driver only invokes and validates the C++ pass output.  Rewrite actions,
legality, and graph identities belong to the pass; Python reference models
are kept in separate modules for numeric regression evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_OPT = Path(
    os.environ.get("AMOEBA_OPT", "mlir-amoeba-opt"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    optimizer = args.amoeba_opt
    if not source.is_file():
        raise ValueError("canonical Taskflow source does not exist")
    if args.max_depth < 0 or args.max_actions <= 0 or args.tile_factor < 0:
        raise ValueError("rewrite budgets and tile factor are invalid")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "joint_variant_space.json"
    # The pass writes a temporary file and publishes only a complete manifest.
    command = [
        str(optimizer), str(source), "--verify-each",
        ("--enumerate-joint-semantic-rewrites="
         f"output={manifest_path} max-depth={args.max_depth} "
         f"max-actions={args.max_actions} tile-factor={args.tile_factor}"),
        "-o", "/dev/null",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("MLIR joint rewrite pass failed: " + result.stderr[-2000:])
    if not manifest_path.is_file():
        raise ValueError("MLIR joint rewrite pass did not publish a manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True or
            not isinstance(manifest.get("candidates"), list)):
        raise ValueError("MLIR joint rewrite pass published an incomplete manifest")
    return {"schema": manifest.get("schema"),
            "complete": True,
            "candidate_count": len(manifest["candidates"]),
            "statistics": manifest.get("statistics"),
            "manifest_path": str(manifest_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amoeba-opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=50000)
    parser.add_argument("--tile-factor", type=int, default=0,
                        help="0 enumerates independent M/N/K factors 2, 4, 8")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parse_args(argv)), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"joint MLIR graph enumeration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
