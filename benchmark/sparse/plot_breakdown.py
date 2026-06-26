#!/usr/bin/env python3
"""Plot solver time breakdown from sparse_benchmark.json.
Two pages: linear scale (page 1), semilogy (page 2).
"""

import json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

HERE = Path(__file__).parent

SECTIONS = ["allocation", "sketch", "matvec", "overlap", "ortho", "diag", "restart", "other"]

# Colorblind-friendly palette (Wong 2011) for named sections; gray for "other"
_wong = [
    "#0072B2",  # blue        – allocation
    "#009E73",  # green       – sketch
    "#D55E00",  # vermillion  – matvec
    "#CC79A7",  # pink        – overlap
    "#56B4E9",  # sky blue    – ortho
    "#E69F00",  # orange      – diag
    "#F0E442",  # yellow      – restart
]
COLORS = {s: _wong[i] for i, s in enumerate(SECTIONS[:-1])}
COLORS["other"] = "#999999"

infile = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'sparse_benchmark.json'
d = json.loads(infile.read_text())
records, k = d["data"], d["k"]

ns = [r["n"] for r in records]
x  = np.arange(len(ns))
w  = 0.35


def build_stacked(records):
    """Return per-section arrays (list of (section, jd_times, sk_times))."""
    result = []
    for s in SECTIONS:
        jd_vals, sk_vals = [], []
        for rec in records:
            sec = dict(rec["sections"])
            jd_sum = sum(v for nm, v in sec.items()
                         if nm.startswith("jd:") and not nm.startswith("jd_sketched:"))
            sk_sum = sum(v for nm, v in sec.items() if nm.startswith("jd_sketched:"))
            sec["jd: other"]          = max(0.0, rec["jd_mean"]          - jd_sum)
            sec["jd_sketched: other"] = max(0.0, rec["jd_sketched_mean"] - sk_sum)
            jd_vals.append(sec.get(f"jd: {s}",          0.0))
            sk_vals.append(sec.get(f"jd_sketched: {s}", 0.0))
        result.append((s, np.array(jd_vals), np.array(sk_vals)))
    return result


def draw_bars(ax, stacked):
    used = set()
    jd_bot = np.zeros(len(records))
    sk_bot = np.zeros(len(records))
    for s, jt, st in stacked:
        c = COLORS[s]
        mask_j = jt > 0
        mask_s = st > 0
        if mask_j.any():
            ax.bar(x[mask_j] - w/2, jt[mask_j], w, bottom=jd_bot[mask_j], color=c)
            jd_bot[mask_j] += jt[mask_j]; used.add(s)
        if mask_s.any():
            ax.bar(x[mask_s] + w/2, st[mask_s], w, bottom=sk_bot[mask_s], color=c)
            sk_bot[mask_s] += st[mask_s]; used.add(s)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['n']:,}" for r in records], rotation=45, ha="right")
    ax.set_xlabel("n   (left bar: Davidson,  right bar: Randomized-Davidson)")
    ax.set_ylabel("Mean solve time [s]")
    ax.set_title(f"Solver time breakdown — k={k}")
    ax.legend(
        handles=[mpatches.Patch(color=COLORS[s], label=s) for s in SECTIONS if s in used],
        loc="upper left", fontsize=8,
    )


def draw_fractions(ax, stacked):
    """Both bars normalized by Davidson time: Davidson=1 by construction,
    Randomized-Davidson can be above or below 1."""
    jd_total = np.array([r["jd_mean"] for r in records])
    used = set()
    jd_bot = np.zeros(len(records))
    sk_bot = np.zeros(len(records))
    for s, jt, st in stacked:
        c   = COLORS[s]
        jf  = jt / jd_total
        sf  = st / jd_total
        if jf.any():
            ax.bar(x - w/2, jf, w, bottom=jd_bot, color=c)
            jd_bot += jf; used.add(s)
        if sf.any():
            ax.bar(x + w/2, sf, w, bottom=sk_bot, color=c)
            sk_bot += sf; used.add(s)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['n']:,}" for r in records], rotation=45, ha="right")
    ax.set_xlabel("n   (left bar: Davidson,  right bar: Randomized-Davidson)")
    ax.set_ylabel("Time relative to Davidson")
    ax.set_title(f"Relative time breakdown — k={k}")
    ax.legend(
        handles=[mpatches.Patch(color=COLORS[s], label=s) for s in SECTIONS if s in used],
        loc="upper left", fontsize=8,
    )


stacked = build_stacked(records)
figsize = (max(8, 1.4 * len(ns)), 6)

outfile = HERE / "sparse_breakdown_from_json.pdf"
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(figsize=figsize)
    draw_bars(ax, stacked)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    draw_fractions(ax, stacked)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved {outfile}  (2 pages: absolute + normalized)")
