#!/usr/bin/env python3
"""
Extract NVTX timing from an nsys report and produce the same JSON schema
as run_benchmark.jl (CPU runs).

Usage:
    python nsys_to_json.py --report report1.nsys-rep \
                           --meta   si_jd_gpu_meta.json \
                           --output results/si_jd_gpu.json

Requires: nsys on PATH (for the SQLite export step).

How it works:
  1. nsys export converts report1.nsys-rep to a SQLite database.
  2. We query the NVTX_EVENTS table for the RandESC timing ranges.
  3. Timing is summed across all calls (all SCF iterations, all k-points).

Note on GPU timing accuracy:
  The NVTX ranges wrap CPU-side launches of GPU kernels. For sections where
  the CPU blocks until the GPU finishes (e.g. because the next operation reads
  the result), this is accurate. For fully async sections it may underestimate
  actual GPU execution time. Use the full nsys timeline for kernel-level detail.
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

NS_TO_S = 1e-9

# NVTX range name → JSON field
JD_RANGES = {
    "jd":           "eigensolver_time_s",
    "jd: matvec":   "matvec_time_s",
    "jd: ortho":    "ortho_time_s",
    "jd: diag":     "rr_time_s",
    "jd: residual": "residual_time_s",
}
JD_SKETCHED_RANGES = {
    "jd_sketched":           "eigensolver_time_s",
    "jd_sketched: matvec":   "matvec_time_s",
    "jd_sketched: ortho":    "ortho_time_s",
    "jd_sketched: diag":     "rr_time_s",
    "jd_sketched: residual": "residual_time_s",
}


def export_to_sqlite(report_path: Path, out_dir: Path) -> Path:
    if report_path.suffix == ".qdstrm":
        print(
            f"ERROR: {report_path.name} is a raw capture file. The nsys importer is\n"
            "not available on this host. Copy it to a machine with the full Nsight\n"
            "Systems installation and run:\n"
            f"  nsys export --type sqlite {report_path.stem}.nsys-rep\n"
            "or just pass the converted .nsys-rep to this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    sqlite_path = out_dir / (report_path.stem + ".sqlite")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite",
         "--output", str(sqlite_path), str(report_path)],
        check=True, capture_output=True,
    )
    return sqlite_path


def query_nvtx(sqlite_path: Path, ranges: dict[str, str]) -> dict[str, float]:
    """Return {json_field: total_seconds} for each requested NVTX range."""
    conn = sqlite3.connect(sqlite_path)
    try:
        # NVTX push/pop ranges are stored as paired events; newer nsys versions
        # store them as single rows with both start and end filled in.
        # We try the single-row format first, then fall back to paired events.
        timings = {}
        for range_name, field in ranges.items():
            total_ns = _query_range(conn, range_name)
            if total_ns is not None:
                timings[field] = total_ns * NS_TO_S
        return timings
    finally:
        conn.close()


def _query_range(conn: sqlite3.Connection, name: str) -> int | None:
    """Return total nanoseconds for a named NVTX range, or None if not found."""

    # Try single-row format (nsys >= 2023): NVTX_EVENTS has start and end
    try:
        rows = conn.execute("""
            SELECT SUM(e.end - e.start)
            FROM   NVTX_EVENTS e
            JOIN   StringIds   s ON e.text = s.id
            WHERE  s.value = ?
              AND  e.end IS NOT NULL
        """, (name,)).fetchone()
        if rows and rows[0] is not None:
            return int(rows[0])
    except sqlite3.OperationalError:
        pass

    # Fallback: text stored inline (not via StringIds)
    try:
        rows = conn.execute("""
            SELECT SUM(end - start)
            FROM   NVTX_EVENTS
            WHERE  text = ?
              AND  end IS NOT NULL
        """, (name,)).fetchone()
        if rows and rows[0] is not None:
            return int(rows[0])
    except sqlite3.OperationalError:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Convert nsys report to benchmark JSON.")
    parser.add_argument("--report", required=True, help="Path to report1.nsys-rep")
    parser.add_argument("--meta",   required=True, help="Path to *_gpu_meta.json written by run_gpu_profile.jl")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    report_path = Path(args.report)
    meta_path   = Path(args.meta)
    output_path = Path(args.output)

    with open(meta_path) as f:
        meta = json.load(f)

    solver = meta.get("solver", "")
    ranges = JD_SKETCHED_RANGES if solver == "jd_sketched" else JD_RANGES

    print(f"Exporting {report_path} to SQLite...")
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = export_to_sqlite(report_path, Path(tmp))
        print(f"Querying NVTX ranges...")
        timings = query_nvtx(sqlite_path, ranges)

    if not timings:
        print("ERROR: no NVTX ranges found. Check that NVTX.jl was loaded and "
              "the report was captured with 'nsys launch'.", file=sys.stderr)
        sys.exit(1)

    result = {
        **meta,
        "wall_time_s":        None,   # not available without DFTK NVTX markers
        "eigensolver_time_s": timings.get("eigensolver_time_s"),
        "matvec_time_s":      timings.get("matvec_time_s"),
        "ortho_time_s":       timings.get("ortho_time_s"),
        "rr_time_s":          timings.get("rr_time_s"),
        "residual_time_s":    timings.get("residual_time_s"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
