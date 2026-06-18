#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description="Plot timings from benchmark CSV")
parser.add_argument("csv", nargs="?", default="results/timings.csv")
parser.add_argument("-o", "--output", default=None)
args = parser.parse_args()

base = os.path.splitext(args.csv)[0]
scatter_out = args.output if args.output else base + "_scatter.pdf"
bar_out = base + "_bar.pdf"

df = pd.read_csv(args.csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

TIMING_COLS   = ["total_scf", "eigensolver_total", "matvec", "diag", "ortho", "allocation"]
SCATTER_COLS  = ["total_scf", "eigensolver_total", "matvec", "diag", "ortho"]
ALGO_ORDER  = ["lobpcg", "jd", "jd_sketched"]
ALGO_LABELS = {"lobpcg": "LOBPCG", "jd": "JD", "jd_sketched": "rand-JD"}
ALGO_PAIRS  = [("lobpcg", "jd"), ("lobpcg", "jd_sketched"), ("jd", "jd_sketched")]

STACK_COLS   = ["matvec", "diag", "ortho", "allocation", "other_eigensolver", "non_eigensolver"]
STACK_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#b0b0b0", "#e8e8e8"]
STACK_LABELS = ["matvec", "diag", "ortho", "allocation", "other eigensolver", "non-eigensolver"]

for col in TIMING_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# n_planewaves is the same for every algorithm on a given structure
npw_per_struct = df.groupby("structure")["n_planewaves"].first()

with PdfPages(scatter_out) as pdf:

    # ------------------------------------------------------------------
    # Pages 1–6: per-section scatter plots
    # ------------------------------------------------------------------
    for col in SCATTER_COLS:
        pivot = df.pivot_table(index="structure", columns="algorithm", values=col, aggfunc="mean")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        fig.suptitle(f"Per-structure comparison: {col} (s)", fontsize=13, fontweight="bold")

        structs = pivot.index
        npw_vals = npw_per_struct.reindex(structs)
        vmin, vmax = npw_vals.min(), npw_vals.max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.viridis

        any_data = False
        for ax, (algo_x, algo_y) in zip(axes, ALGO_PAIRS):
            ax.set_title(f"{ALGO_LABELS[algo_x]} vs {ALGO_LABELS[algo_y]}")

            if algo_x not in pivot.columns or algo_y not in pivot.columns:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        color="gray")
                continue

            sub = pivot[[algo_x, algo_y]].dropna()
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        color="gray")
                continue

            any_data = True
            c_vals = npw_per_struct.reindex(sub.index).values

            ax.scatter(sub[algo_x], sub[algo_y], c=c_vals, cmap=cmap, norm=norm,
                       s=28, alpha=0.85, linewidths=0)

            lo = min(sub[algo_x].min(), sub[algo_y].min())
            hi = max(sub[algo_x].max(), sub[algo_y].max())
            pad = (hi - lo) * 0.05 or hi * 0.05
            lims = (lo - pad, hi + pad)
            ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"{ALGO_LABELS[algo_x]} (s)")
            ax.set_ylabel(f"{ALGO_LABELS[algo_y]} (s)")

        if any_data:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.75, pad=0.02)
            cbar.set_label("n_planewaves")

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved {scatter_out}  ({len(SCATTER_COLS)} pages)")

# ------------------------------------------------------------------
# Stacked bar chart of means — separate file
# ------------------------------------------------------------------
means = df.groupby("algorithm")[TIMING_COLS].mean().reindex(ALGO_ORDER)

# Build per-algo stacks
stack = pd.DataFrame(index=ALGO_ORDER)
for c in ["matvec", "diag", "ortho"]:
    stack[c] = means[c].fillna(0)
stack["allocation"] = means["allocation"].fillna(0)
stack["other_eigensolver"] = (
    means["eigensolver_total"]
    - stack["matvec"] - stack["diag"] - stack["ortho"] - stack["allocation"]
).clip(lower=0)
stack["non_eigensolver"] = (
    means["total_scf"] - means["eigensolver_total"]
).clip(lower=0)

fig, ax = plt.subplots(figsize=(7, 5))
x_pos = np.arange(len(ALGO_ORDER))
bottom = np.zeros(len(ALGO_ORDER))

for col_name, color, label in zip(STACK_COLS, STACK_COLORS, STACK_LABELS):
    vals = stack[col_name].values
    ax.bar(x_pos, vals, bottom=bottom, color=color, label=label,
           edgecolor="white", linewidth=0.5, width=0.55)
    bottom += vals

ax.set_xticks(x_pos)
ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER])
ax.set_ylabel("Mean time (s)")
ax.set_title("Mean timing breakdown per algorithm")
ax.set_ylim(bottom=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(bar_out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {bar_out}")
