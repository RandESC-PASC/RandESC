#!/usr/bin/env python3
"""Parse slurm-12912.out (or any sparse.jl output) into sparse_benchmark.json."""

import re, json, sys
from pathlib import Path

HERE = Path(__file__).parent


def parse_time(s):
    s = s.strip()
    if s.endswith('ns'):         return float(s[:-2]) * 1e-9
    if 'μs' in s or 'µs' in s:  return float(re.sub(r'[μµ]s', '', s)) * 1e-6
    if s.endswith('ms'):         return float(s[:-2]) * 1e-3
    if s.endswith('h'):          return float(s[:-1]) * 3600.0
    if s.endswith('s'):          return float(s[:-1])
    raise ValueError(f"Unknown time format: {s!r}")


SUMMARY_RE = re.compile(
    r'n=(\d+)\s+\(requested \d+\).*?'
    r'jd=([\d.]+)[±+]([\d.]+)s\s+'
    r'jd_sketched=([\d.]+)[±+]([\d.]+)s'
)
# Matches indented sub-section lines like:
#   jd: ortho                     66    6.22s   ...
#   jd_sketched: restart          90    1479s   ...
SECTION_RE = re.compile(r'^\s+(jd(?:_sketched)?:\s+\w+)\s+\d+\s+(\S+)')
K_RE       = re.compile(r'=== k = (\d+) ===')
OP_RE      = re.compile(r'^Operator:\s+(\S+)')

infile = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'slurm-12912.out'
N_REPS = 3

records, pending = [], None
current_k, operator, in_timer = 500, 'unknown', False

for line in infile.read_text(encoding='utf-8').splitlines():
    m = K_RE.search(line)
    if m:
        current_k = int(m.group(1)); continue

    m = OP_RE.match(line)
    if m:
        operator = m.group(1); continue

    m = SUMMARY_RE.search(line)
    if m:
        if pending is not None:
            records.append(pending)
        pending = dict(
            n=int(m.group(1)),
            jd_mean=float(m.group(2)),          jd_std=float(m.group(3)),
            jd_sketched_mean=float(m.group(4)), jd_sketched_std=float(m.group(5)),
            sections={},
        )
        in_timer = False
        continue

    if '── RandESC timer' in line:
        in_timer = True; continue

    if in_timer and pending is not None:
        m = SECTION_RE.match(line)
        if m:
            try:
                pending['sections'][m.group(1).strip()] = parse_time(m.group(2)) / N_REPS
            except ValueError:
                pass

if pending is not None:
    records.append(pending)

out = dict(k=current_k, n_reps=N_REPS, operator=operator, data=records)
outfile = HERE / 'sparse_benchmark.json'
outfile.write_text(json.dumps(out, indent=2))
print(f"Wrote {outfile}  ({len(records)} data points)")
