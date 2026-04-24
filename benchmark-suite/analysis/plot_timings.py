#!/usr/bin/env python3
"""
Plot benchmark timing comparisons across solvers.

Usage:
    python plot_timings.py --results <results_dir> [--output <dir|file>] [--system <name>]

Produces two figures per system:
  - Bar chart comparing all timing fields across solvers
  - Pie charts showing the wall-time breakdown per solver
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
    ("residual_time_s",    "Residual"),
]

SOLVER_ORDER  = ["lobpcg", "jd", "jd_sketched"]
SOLVER_LABELS = {"lobpcg": "LOBPCG", "jd": "JD", "jd_sketched": "JD (sketched)"}
SOLVER_COLORS = ["#4c72b0", "#dd8452", "#55a868"]

PIE_LABELS = ["Matvec", "Ortho", "Rayleigh-Ritz", "Residual", "Other eigensolver"]
PIE_COLORS = ["#e07070", "#70a0e0", "#70c870", "#a070c8", "#c8a870"]


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


def make_output_path(output_arg: Path | None, system: str, n_systems: int, suffix: str) -> Path | None:
    if output_arg is None:
        return None
    if output_arg.is_dir() or (n_systems > 1 and output_arg.suffix == ""):
        output_arg.mkdir(parents=True, exist_ok=True)
        return output_arg / f"{system}_{suffix}.pdf"
    stem = output_arg.stem
    ext  = output_arg.suffix or ".pdf"
    return output_arg.parent / f"{stem}_{system}_{suffix}{ext}"


def solver_info_line(solver: str, r: dict) -> str:
    parts = [SOLVER_LABELS.get(solver, solver)]
    if r.get("n_matvec")  is not None: parts.append(f"matvec={r['n_matvec']}")
    if r.get("n_scf_iter") is not None: parts.append(f"SCF iter={r['n_scf_iter']}")
    parts.append("converged" if r.get("converged") else "NOT CONVERGED")
    return "  |  ".join(parts)


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_bar(system: str, solver_data: dict[str, dict], output_path: Path | None):
    solvers = [s for s in SOLVER_ORDER if s in solver_data]
    if not solvers:
        return

    fields    = [(k, label) for k, label in TIMING_FIELDS
                 if any(solver_data[s].get(k) for s in solvers)]
    n_fields  = len(fields)
    n_solvers = len(solvers)
    x         = np.arange(n_fields)
    width     = 0.8 / n_solvers

    fig, ax = plt.subplots(figsize=(max(8, n_fields * 2), 5))

    for i, solver in enumerate(solvers):
        vals   = [solver_data[solver].get(k, 0) or 0 for k, _ in fields]
        offset = (i - n_solvers / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=SOLVER_LABELS.get(solver, solver),
                        color=SOLVER_COLORS[i % len(SOLVER_COLORS)], alpha=0.85)
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

    info = "\n".join(solver_info_line(s, solver_data[s]) for s in solvers)
    fig.text(0.5, -0.04, info, ha="center", va="top", fontsize=8, family="monospace")

    fig.tight_layout()
    _save_or_show(fig, output_path)


# ── Pie charts ────────────────────────────────────────────────────────────────

def _pie_slices(r: dict) -> tuple[list[float], list[str]]:
    """Break eigensolver_time_s into components."""
    eig      = r.get("eigensolver_time_s", 0) or 0
    mv       = r.get("matvec_time_s", 0) or 0
    ortho    = r.get("ortho_time_s", 0) or 0
    rr       = r.get("rr_time_s", 0) or 0
    residual = r.get("residual_time_s", 0) or 0

    other_eig = max(eig - mv - ortho - rr - residual, 0)

    vals   = [mv, ortho, rr, residual, other_eig]
    labels = PIE_LABELS
    # Filter zero slices
    pairs  = [(v, l) for v, l in zip(vals, labels) if v > 0]
    if not pairs:
        return [], []
    vals, labels = zip(*pairs)
    return list(vals), list(labels)


def plot_pie(system: str, solver_data: dict[str, dict], output_path: Path | None):
    solvers = [s for s in SOLVER_ORDER if s in solver_data]
    if not solvers:
        return

    fig, axes = plt.subplots(1, len(solvers),
                              figsize=(4.5 * len(solvers), 5))
    if len(solvers) == 1:
        axes = [axes]

    color_map = {label: color for label, color in zip(PIE_LABELS, PIE_COLORS)}

    for ax, solver in zip(axes, solvers):
        r          = solver_data[solver]
        vals, lbls = _pie_slices(r)
        if not vals:
            ax.set_visible(False)
            continue

        colors = [color_map.get(l, "#aaaaaa") for l in lbls]
        _, _, autotexts = ax.pie(
            vals, labels=None, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)

        eig = r.get("eigensolver_time_s", 0) or 0
        ax.set_title(f"{SOLVER_LABELS.get(solver, solver)}\n({eig:.1f}s eigensolver)", fontsize=10)

    # Shared legend using all possible labels
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l]) for l in PIE_LABELS]
    fig.legend(handles, PIE_LABELS, loc="lower center", ncol=len(PIE_LABELS),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f"Wall-time breakdown — {system}", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, output_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, output_path: Path | None):
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot solver timing comparisons.")
    parser.add_argument("--results", required=True, help="Directory with result JSONs")
    parser.add_argument("--output",  default=None,
                        help="Output file or directory. Bar chart gets _bar suffix, "
                             "pie chart gets _pie suffix. Omit for interactive display.")
    parser.add_argument("--system",  default=None, help="Restrict to one system name")
    args = parser.parse_args()

    results_dir = Path(args.results)
    data        = load_results(results_dir, args.system)

    if not data:
        print("No data found.")
        return

    output_arg = Path(args.output) if args.output else None
    n_systems  = len(data)

    for system, solver_data in sorted(data.items()):
        bar_out = make_output_path(output_arg, system, n_systems, "bar")
        pie_out = make_output_path(output_arg, system, n_systems, "pie")
        plot_bar(system, solver_data, bar_out)
        plot_pie(system, solver_data, pie_out)


if __name__ == "__main__":
    main()
