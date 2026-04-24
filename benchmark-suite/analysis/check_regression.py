#!/usr/bin/env python3
"""
Compare benchmark results against reference timings.

Usage:
    python check_regression.py --results <results_dir> --reference <reference_dir> [--tol 0.20]

For each JSON file in results_dir, looks for a matching file in reference_dir
(same name) and checks that no timing field has increased by more than `tol`
(default 20%) relative to the reference.

Exit code: 0 if all checks pass, 1 if any regression is detected.
"""

import argparse
import json
import sys
from pathlib import Path

TIMING_FIELDS = [
    "wall_time_s",
    "eigensolver_time_s",
    "matvec_time_s",
    "ortho_time_s",
    "rr_time_s",
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_file(result_path: Path, ref_path: Path, tol: float) -> list[str]:
    result = load_json(result_path)
    ref = load_json(ref_path)
    failures = []

    for field in TIMING_FIELDS:
        ref_val = ref.get(field)
        new_val = result.get(field)

        if ref_val is None or new_val is None:
            continue
        if ref_val == 0:
            continue

        ratio = new_val / ref_val
        if ratio > 1.0 + tol:
            failures.append(
                f"  {field}: {new_val:.3f}s vs reference {ref_val:.3f}s "
                f"(+{(ratio - 1)*100:.1f}%, limit +{tol*100:.0f}%)"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Check benchmark regressions.")
    parser.add_argument("--results",   required=True, help="Directory with new result JSONs")
    parser.add_argument("--reference", required=True, help="Directory with reference JSONs")
    parser.add_argument("--tol", type=float, default=0.20,
                        help="Allowed relative slowdown (default: 0.20 = 20%%)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    ref_dir     = Path(args.reference)

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    any_regression = False
    any_compared   = False

    for result_path in result_files:
        ref_path = ref_dir / result_path.name
        if not ref_path.exists():
            print(f"[SKIP] {result_path.name} — no reference file found")
            continue

        any_compared = True
        failures = check_file(result_path, ref_path, args.tol)

        result = load_json(result_path)
        label = f"{result.get('system','?')} / {result.get('solver','?')}"

        if failures:
            any_regression = True
            print(f"[FAIL] {label}")
            for msg in failures:
                print(msg)
        else:
            print(f"[PASS] {label}")

    if not any_compared:
        print("No files could be compared (no matching reference files).")
        sys.exit(1)

    sys.exit(1 if any_regression else 0)


if __name__ == "__main__":
    main()
