#!/usr/bin/env python3
"""
Parse `nsys stats --report nvtx_sum` output and write a JSON compatible
with plot_timings.py.

Usage:
    python parse_nvtx.py --report <report.nsys-rep> --solver <jd|jd_sketched>
                         --system <name> --output <out.json>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

NS_TO_S = 1e-9

RANGE_FIELDS = {
    "eigensolver_time_s": lambda s: s,
    "matvec_time_s":      lambda s: f"{s}: matvec",
    "ortho_time_s":       lambda s: f"{s}: ortho",
    "rr_time_s":          lambda s: f"{s}: diag",
    "residual_time_s":    lambda s: f"{s}: residual",
}


def parse_nvtx_stats(text):
    rows = {}
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Time (%)"):
            in_table = True
            continue
        if s.startswith("---"):
            continue
        if not in_table:
            continue
        tokens = s.split()
        if len(tokens) < 10:
            break
        try:
            range_name = " ".join(tokens[9:]).lstrip(":").strip()
            rows[range_name] = int(tokens[1].replace(",", ""))
        except (ValueError, IndexError):
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_sum", args.report],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    print(result.stdout)  # show the table in the terminal

    rows = parse_nvtx_stats(result.stdout)
    if not rows:
        print("No NVTX ranges found.", file=sys.stderr)
        sys.exit(1)

    data = {"solver": args.solver, "system": args.system, "architecture": "gpu"}
    for field, range_fn in RANGE_FIELDS.items():
        ns = rows.get(range_fn(args.solver))
        data[field] = ns * NS_TO_S if ns is not None else None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
