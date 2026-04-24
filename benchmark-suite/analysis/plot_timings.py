#!/usr/bin/env python3
"""
Plot benchmark timing comparisons across solvers.

Usage:
    python plot_timings.py --results <results_dir> [--output <plot.pdf>] [--system <name>]

For each system found in results_dir, produces a grouped bar chart comparing
wall_time_s, eigensolver_time_s, matvec_time_s, ortho_time_s, and rr_time_s
across solvers.

Multiple systems are plotted on separate figures. Pass --system to restrict to one.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

TIMING_FIELDS = [
    ("wall_time_s",        "Wall time"),
    ("eigensolver_time_s", "Eigensolver"),
    ("matvec_time_s",      "Matvec"),
    ("ortho_time_s",       "Ortho"),
    ("rr_time_s",          "Rayleigh-Ritz"),
]

SOLVER_ORDER  = ["lobpcg", "jd", "jd_sketched"]
SOLVER_LABELS = {"lobpcg": "LOBPCG", "jd": "JD", "jd_sketched": "JD (sketched)"}
COLORS        = ["#4c72b0", "#dd8452", "#55a868"]


def load_results(results_dir: Path, system_filter: str | None) -> dict[str, dict[str, dict]]:
    """Returns {system: {solver: result_dict}}."""
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        system = r.get("system", path.stem)
        solver = r.get("solver", "?")
        if system_filter and system != system_filter:
            continue
        data[system][solver] = r
    return dict(data)


def plot_system(system: str, solver_data: dict[str, dict], output_path: Path | None):
    solvers = [s for s in SOLVER_ORDER if s in solver_data]
    if not solvers:
        print(f"[SKIP] {system}: no recognised solvers found")
        return

    fields   = [(k, label) for k, label in TIMING_FIELDS
                if any(solver_data[s].get(k) for s in solvers)]
    n_fields = len(fields)
    n_solvers = len(solvers)

    x      = np.arange(n_fields)
    width  = 0.8 / n_solvers
    fig, ax = plt.subplots(figsize=(max(8, n_fields * 2), 5))

    for i, solver in enumerate(solvers):
        vals = [solver_data[solver].get(k, 0) or 0 for k, _ in fields]
        offset = (i - n_solvers / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=SOLVER_LABELS.get(solver, solver),
                      color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.1f}s", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in fields])
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Solver timing comparison — {system}")
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
    ax.set_axisbelow(True)

    # Annotate n_matvec and n_scf_iter in a small table below the legend
    info_lines = []
    for solver in solvers:
        r = solver_data[solver]
        n_mv  = r.get("n_matvec")
        n_scf = r.get("n_scf_iter")
        conv  = r.get("converged", "?")
        parts = [SOLVER_LABELS.get(solver, solver)]
        if n_mv  is not None: parts.append(f"matvec={n_mv}")
        if n_scf is not None: parts.append(f"SCF iter={n_scf}")
        parts.append("converged" if conv else "NOT CONVERGED")
        info_lines.append("  |  ".join(parts))
    fig.text(0.5, -0.04, "\n".join(info_lines), ha="center", va="top",
             fontsize=8, family="monospace")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot solver timing comparisons.")
    parser.add_argument("--results", required=True, help="Directory with result JSONs")
    parser.add_argument("--output",  default=None,
                        help="Output file (PDF/PNG). If omitted, shows interactive plot. "
                             "For multiple systems, use a directory path.")
    parser.add_argument("--system",  default=None, help="Restrict to one system name")
    args = parser.parse_args()

    results_dir = Path(args.results)
    data = load_results(results_dir, args.system)

    if not data:
        print("No data found.")
        return

    output_arg = Path(args.output) if args.output else None

    for system, solver_data in sorted(data.items()):
        if output_arg is not None:
            if output_arg.is_dir() or (len(data) > 1 and output_arg.suffix == ""):
                output_arg.mkdir(parents=True, exist_ok=True)
                out = output_arg / f"{system}.pdf"
            elif len(data) > 1:
                stem   = output_arg.stem
                suffix = output_arg.suffix
                out    = output_arg.parent / f"{stem}_{system}{suffix}"
            else:
                out = output_arg
        else:
            out = None

        plot_system(system, solver_data, out)


if __name__ == "__main__":
    main()
