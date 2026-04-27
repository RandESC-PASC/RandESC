#!/usr/bin/env python3
"""
Parse `nsys stats --report nvtx_sum` output to JSON.

Usage:
    python parse_nvtx.py <report.nsys-rep> [output.json]

If no output file is given, prints JSON to stdout.
"""

import json
import subprocess
import sys
from pathlib import Path


def parse_nvtx_stats(text):
    rows = []
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
            break  # end of table
        try:
            range_name = " ".join(tokens[9:]).lstrip(":").strip()
            rows.append({
                "range":        range_name,
                "time_pct":     float(tokens[0]),
                "total_time_ns": int(tokens[1].replace(",", "")),
                "instances":    int(tokens[2]),
                "avg_ns":       float(tokens[3].replace(",", "")),
                "med_ns":       float(tokens[4].replace(",", "")),
                "min_ns":       int(tokens[5].replace(",", "")),
                "max_ns":       int(tokens[6].replace(",", "")),
                "stddev_ns":    float(tokens[7].replace(",", "")),
            })
        except (ValueError, IndexError):
            break
    return rows


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <report.nsys-rep> [output.json]", file=sys.stderr)
        sys.exit(1)

    report = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    result = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_sum", str(report)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    rows = parse_nvtx_stats(result.stdout)
    if not rows:
        print("No NVTX ranges found in output.", file=sys.stderr)
        sys.exit(1)

    data = {"report": str(report), "nvtx_ranges": rows}

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2))
        print(f"Written to {output}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
