#!/usr/bin/env python3
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Summarize average runtime and variance per algorithm from timings.csv")
parser.add_argument("csv", nargs="?", default="results/timings.csv")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

TIMING_COLS = ["total_scf", "eigensolver_total", "matvec", "diag", "ortho", "allocation"]

for col in TIMING_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

summary = (
    df.groupby("algorithm")[TIMING_COLS]
    .agg(["mean", "var", "count"])
)

summary.columns = ["_".join(c) for c in summary.columns]

print(f"Loaded {len(df)} rows from {args.csv}\n")
for algo, row in summary.iterrows():
    n = int(row["total_scf_count"])
    print(f"=== {algo} (n={n}) ===")
    for col in TIMING_COLS:
        mean = row[f"{col}_mean"]
        var = row[f"{col}_var"]
        if pd.isna(mean):
            continue
        std = var ** 0.5 if not pd.isna(var) else float("nan")
        print(f"  {col:<22}  mean={mean:8.3f}s  std={std:7.3f}s")
    print()
