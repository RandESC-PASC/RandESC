#!/usr/bin/env python3
"""Histogram of average SCF timing vs. number of plane waves for jd and jd_sketched.

Each bar shows the mean total SCF time (transparent) with the mean solver-only
time overlaid (darker, less transparent). If DFTK output files are available, a
second figure shows the same breakdown per SCF step.
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Display labels: the solver in the code is JD, but publications label it GD.
ALGO_LABELS = {"jd": "GD", "jd_sketched": "sketched-GD"}
ALGO_COLORS = {"jd": "#4c72b0", "jd_sketched": "#55a868"}
TOTAL_ALPHA = 0.35
SOLVER_ALPHA = 0.95

# Larger caption sizes.
TITLE_SIZE = 16
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 12


def infer_outputs_dir(csv_path):
    """Infer outputs/ directory from a results/ sibling, e.g. results_gpu_1 -> outputs_gpu_1."""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.basename(csv_dir)
    if base.startswith("results_"):
        sibling = "outputs_" + base[len("results_"):]
        candidate = os.path.join(os.path.dirname(csv_dir), sibling)
        if os.path.isdir(candidate):
            return candidate
    return None


def parse_scf_steps(out_path):
    """Parse a DFTK .out file and return {algorithm: n_scf_steps}.

    Maps LOBPCG -> lobpcg, JD -> jd, rand-JD -> jd_sketched.
    """
    section_map = {"LOBPCG": "lobpcg", "JD": "jd", "rand-JD": "jd_sketched"}
    steps = {}

    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in section_map:
            algo = section_map[stripped]
            i += 1
            # Skip blank line after section title.
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            # Skip table header.
            if i < len(lines) and lines[i].strip().startswith("n"):
                i += 1
            # Skip header separator.
            if i < len(lines) and lines[i].strip().startswith("---"):
                i += 1

            count = 0
            while i < len(lines):
                line = lines[i]
                # Stop at the box-drawing separator after the iteration table.
                if line.startswith("\u2500"):
                    break
                if re.match(r"^\s*\d+\s", line):
                    count += 1
                    i += 1
                else:
                    break
            steps[algo] = count
        i += 1

    return steps


def collect_scf_steps(csv_df, outputs_dir):
    """Return a DataFrame with algorithm/structure/n_scf_steps from .out files."""
    rows = []
    for structure in csv_df["structure"].unique():
        out_path = os.path.join(outputs_dir, f"{structure}.out")
        if not os.path.isfile(out_path):
            continue
        try:
            steps = parse_scf_steps(out_path)
        except Exception:
            continue
        for algo, n in steps.items():
            rows.append({"structure": structure, "algorithm": algo, "n_scf_steps": n})
    return pd.DataFrame(rows)


def bin_and_mean(npw_vals, jd_vals, sk_vals, edges):
    """Return (jd_means, sk_means, counts) for the given bins."""
    n_bins = len(edges) - 1
    jd_means = np.full(n_bins, np.nan)
    sk_means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (npw_vals >= lo) & (npw_vals < hi)
        else:
            mask = (npw_vals >= lo) & (npw_vals <= hi)
        counts[i] = mask.sum()
        if counts[i] > 0:
            jd_means[i] = jd_vals[mask].mean()
            sk_means[i] = sk_vals[mask].mean()

    return jd_means, sk_means, counts


def make_labels(edges, counts):
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"{lo/1e3:.0f}k–{hi/1e3:.0f}k\n(n={counts[i]})")
    return labels


def draw_overlay_bars(ax, x, jd_total, jd_solver, sk_total, sk_solver, width):
    """Draw transparent total bars with a narrower darker solver-time overlay."""
    solver_width = 0.80 * width

    # Total SCF time (transparent background with white outline).
    ax.bar(
        x - width / 2,
        np.where(np.isnan(jd_total), 0, jd_total),
        width,
        color=ALGO_COLORS["jd"],
        alpha=TOTAL_ALPHA,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.bar(
        x + width / 2,
        np.where(np.isnan(sk_total), 0, sk_total),
        width,
        color=ALGO_COLORS["jd_sketched"],
        alpha=TOTAL_ALPHA,
        edgecolor="white",
        linewidth=0.7,
    )
    # Solver-only time (darker, narrower overlay).
    ax.bar(
        x - width / 2,
        np.where(np.isnan(jd_solver), 0, jd_solver),
        solver_width,
        color=ALGO_COLORS["jd"],
        alpha=SOLVER_ALPHA,
    )
    ax.bar(
        x + width / 2,
        np.where(np.isnan(sk_solver), 0, sk_solver),
        solver_width,
        color=ALGO_COLORS["jd_sketched"],
        alpha=SOLVER_ALPHA,
    )


def make_legend():
    return [
        Patch(facecolor=ALGO_COLORS["jd"], alpha=TOTAL_ALPHA, label=f"{ALGO_LABELS['jd']} total"),
        Patch(facecolor=ALGO_COLORS["jd"], alpha=SOLVER_ALPHA, label=f"{ALGO_LABELS['jd']} solver"),
        Patch(
            facecolor=ALGO_COLORS["jd_sketched"],
            alpha=TOTAL_ALPHA,
            label=f"{ALGO_LABELS['jd_sketched']} total",
        ),
        Patch(
            facecolor=ALGO_COLORS["jd_sketched"],
            alpha=SOLVER_ALPHA,
            label=f"{ALGO_LABELS['jd_sketched']} solver",
        ),
    ]


def draw_one_figure(
    title,
    ylabel,
    labels,
    jd_total,
    sk_total,
    jd_solver,
    sk_solver,
):
    """Create and return a single histogram figure."""
    # Flatter, wider aspect ratio.
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35

    draw_overlay_bars(
        ax, x, jd_total, jd_solver, sk_total, sk_solver, width
    )

    ax.set_xlabel("Number of plane waves\n(# structures)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.legend(handles=make_legend(), ncol=2, fontsize=LEGEND_SIZE)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Histogram of average SCF timing per plane-wave bin"
    )
    parser.add_argument("csv", nargs="?", default="results/timings.csv")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output base name; '_total.pdf' and '_per_step.pdf' are appended",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="number of linear plane-wave bins (default: 5)",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="directory containing DFTK .out files (auto-detected by default)",
    )
    args = parser.parse_args()

    csv_base = os.path.splitext(args.csv)[0]
    if args.output:
        out_base = os.path.splitext(args.output)[0]
    else:
        out_base = csv_base + "_histogram"

    total_path = out_base + "_total.pdf"
    per_step_path = out_base + "_per_step.pdf"

    outputs_dir = args.outputs_dir
    if outputs_dir is None:
        outputs_dir = infer_outputs_dir(args.csv)

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["total_scf"] = pd.to_numeric(df["total_scf"], errors="coerce")
    df["eigensolver_total"] = pd.to_numeric(df["eigensolver_total"], errors="coerce")
    df["n_planewaves"] = pd.to_numeric(df["n_planewaves"], errors="coerce")

    # Keep only the two algorithms of interest.
    df = df[df["algorithm"].isin(["jd", "jd_sketched"])]

    # Pivot so each structure has both timings available.
    pivot = df.pivot_table(
        index="structure",
        columns="algorithm",
        values=["total_scf", "eigensolver_total", "n_planewaves"],
        aggfunc="first",
    )

    total = pivot["total_scf"]
    solver = pivot["eigensolver_total"]
    npw = pivot["n_planewaves"]["jd"].combine_first(
        pivot["n_planewaves"]["jd_sketched"]
    )

    valid = (
        total[["jd", "jd_sketched"]].notna().all(axis=1)
        & solver[["jd", "jd_sketched"]].notna().all(axis=1)
        & npw.notna()
    )
    jd_total = total.loc[valid, "jd"].to_numpy()
    sk_total = total.loc[valid, "jd_sketched"].to_numpy()
    jd_solver = solver.loc[valid, "jd"].to_numpy()
    sk_solver = solver.loc[valid, "jd_sketched"].to_numpy()
    npw_vals = npw.loc[valid].to_numpy()

    if len(npw_vals) == 0:
        print("No structures with both jd and jd_sketched data found.")
        return

    # Linear bin edges.
    min_npw, max_npw = npw_vals.min(), npw_vals.max()
    edges = np.linspace(min_npw, max_npw, args.bins + 1)

    jd_total_mean, sk_total_mean, counts = bin_and_mean(
        npw_vals, jd_total, sk_total, edges
    )
    jd_solver_mean, sk_solver_mean, _ = bin_and_mean(
        npw_vals, jd_solver, sk_solver, edges
    )
    labels = make_labels(edges, counts)

    # Total timing figure.
    fig = draw_one_figure(
        title="Mean total timing breakdown with MPC pseudopotentials",
        ylabel="Mean time (s)",
        labels=labels,
        jd_total=jd_total_mean,
        sk_total=sk_total_mean,
        jd_solver=jd_solver_mean,
        sk_solver=sk_solver_mean,
    )
    fig.savefig(total_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {total_path} ({len(npw_vals)} structures, {args.bins} bins)")

    # Per-SCF-step figure (if output files are available).
    per_step_available = False
    if outputs_dir is not None and os.path.isdir(outputs_dir):
        steps_df = collect_scf_steps(df, outputs_dir)
        if not steps_df.empty:
            merged = df.merge(steps_df, on=["structure", "algorithm"], how="left")
            merged = merged[merged["n_scf_steps"] > 0].copy()
            merged["total_per_step"] = merged["total_scf"] / merged["n_scf_steps"]
            merged["solver_per_step"] = (
                merged["eigensolver_total"] / merged["n_scf_steps"]
            )

            per_step_pivot = merged.pivot_table(
                index="structure",
                columns="algorithm",
                values=["total_per_step", "solver_per_step"],
                aggfunc="first",
            )
            per_step_pivot = per_step_pivot.reindex(valid.index)

            ps_valid = (
                per_step_pivot["total_per_step"][["jd", "jd_sketched"]]
                .notna()
                .all(axis=1)
                & per_step_pivot["solver_per_step"][["jd", "jd_sketched"]]
                .notna()
                .all(axis=1)
            )
            if ps_valid.sum() > 0:
                ps_npw = npw.loc[ps_valid].to_numpy()
                ps_jd_total = per_step_pivot["total_per_step"].loc[
                    ps_valid, "jd"
                ].to_numpy()
                ps_sk_total = per_step_pivot["total_per_step"].loc[
                    ps_valid, "jd_sketched"
                ].to_numpy()
                ps_jd_solver = per_step_pivot["solver_per_step"].loc[
                    ps_valid, "jd"
                ].to_numpy()
                ps_sk_solver = per_step_pivot["solver_per_step"].loc[
                    ps_valid, "jd_sketched"
                ].to_numpy()

                jd_total_per_step, sk_total_per_step, _ = bin_and_mean(
                    ps_npw, ps_jd_total, ps_sk_total, edges
                )
                jd_solver_per_step, sk_solver_per_step, _ = bin_and_mean(
                    ps_npw, ps_jd_solver, ps_sk_solver, edges
                )

                fig = draw_one_figure(
                    title="Mean timing breakdown per SCF step with MPC pseudopotentials",
                    ylabel="Time per step (s)",
                    labels=labels,
                    jd_total=jd_total_per_step,
                    sk_total=sk_total_per_step,
                    jd_solver=jd_solver_per_step,
                    sk_solver=sk_solver_per_step,
                )
                fig.savefig(per_step_path, bbox_inches="tight")
                plt.close(fig)
                per_step_available = True
                print(f"Saved {per_step_path}")

    print("Bin counts:", counts.tolist())
    if per_step_available:
        print("SCF-step figure included (output files parsed from " + outputs_dir + ")")
    elif outputs_dir is None:
        print("SCF-step figure skipped (no outputs directory found)")
    else:
        print("SCF-step figure skipped (no usable .out files found in " + outputs_dir + ")")


if __name__ == "__main__":
    main()
