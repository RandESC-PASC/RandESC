#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="results/timings.csv")
parser.add_argument("-o", "--output", default=None, help="output PDF path")
args = parser.parse_args()

CSV_PATH = args.csv

ALGOS  = ["lobpcg", "jd", "jd_sketched"]
LABELS = {"lobpcg": "LOBPCG", "jd": "JD", "jd_sketched": "rand-JD"}
CATS   = ["matvec", "diag", "ortho", "allocation", "other_eigensolver", "non_eigensolver"]
COLORS = {"matvec": "#4C72B0", "diag": "#DD8452", "ortho": "#55A868",
          "allocation": "#C44E52", "other_eigensolver": "#B0B0B0", "non_eigensolver": "#E8E8E8"}

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Keep the latest row per algorithm
df = df.sort_values("timestamp").groupby("algorithm").last().reset_index()

rows = {row["algorithm"]: row for _, row in df.iterrows()}

fig, ax = plt.subplots(figsize=(5, 5))
x = np.arange(len(ALGOS))
bottoms = np.zeros(len(ALGOS))

for cat in CATS:
    heights = []
    for algo in ALGOS:
        row = rows.get(algo)
        if row is None:
            heights.append(0.0)
            continue
        if cat == "other_eigensolver":
            known = sum(float(row[c]) for c in ["matvec", "diag", "ortho", "allocation"]
                        if pd.notna(row.get(c)))
            heights.append(max(0.0, float(row["eigensolver_total"]) - known))
        elif cat == "non_eigensolver":
            heights.append(max(0.0, float(row["total_scf"]) - float(row["eigensolver_total"])))
        else:
            v = row.get(cat)
            heights.append(float(v) if pd.notna(v) else 0.0)
    ax.bar(x, heights, 0.6, bottom=bottoms, color=COLORS[cat], label=cat)
    bottoms += np.array(heights)

r = df.iloc[0]
ax.set_xticks(x)
ax.set_xticklabels([LABELS[a] for a in ALGOS])
ax.set_ylabel("Time (s)")
ax.set_title(f"{r['structure']}  ecut={r['ecut']}  kgrid={r['kgrid']}  gpu={r['use_gpu']}")
ax.set_ylim(bottom=0)

legend_handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CATS]
ax.legend(handles=legend_handles, framealpha=0.9)
plt.tight_layout()

out = args.output if args.output else os.path.splitext(CSV_PATH)[0] + ".pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
