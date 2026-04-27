#!/usr/bin/env bash
#SBATCH --job-name=randesc-gpu-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-gpu-%j.out
#SBATCH --error=slurm-gpu-%j.err
#
# Produces report1.nsys-rep in the working directory.
# View locally with: nsys-ui report1.nsys-rep
# Stats with:        nsys stats --report nvtx_sum report1.nsys-rep
#
# Submit with:
#   sbatch submit_slurm_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────

STRUCTURE="../example/DFTK_solver/h3s.extxyz"
SOLVER="jd"           # jd or jd_sketched
ECUT=30
KGRID="2,2,2"
SCF_MAXITER=10        # keep low to avoid huge nsys dumps

SYSTEM="$(basename "${STRUCTURE%.*}")"

# ── Module / environment setup ────────────────────────────────────────────────
# Uncomment and adapt for your cluster:
# module load julia/1.11 cuda/12

# ── Run ───────────────────────────────────────────────────────────────────────

REPORT_STEM="${SYSTEM}_${SOLVER}_gpu"
META_FILE="${REPORT_STEM}_meta.json"

nsys profile \
    --capture-range=cudaProfilerApi \
    --force-overwrite true \
    --output "${REPORT_STEM}" \
    julia \
        --project="$SCRIPT_DIR" \
        --threads=1 \
        "$SCRIPT_DIR/run_gpu_profile.jl" \
        "$STRUCTURE" \
        --solver "$SOLVER" \
        --ecut "$ECUT" \
        --kgrid "$KGRID" \
        --scf-maxiter "$SCF_MAXITER" || true  # nsys exits non-zero after SIGTERM

# Locate the report: nsys writes <stem>.nsys-rep in CWD when the importer is
# available, otherwise falls back to a .qdstrm in /tmp.
if [[ -f "${REPORT_STEM}.nsys-rep" ]]; then
    REPORT="${REPORT_STEM}.nsys-rep"
elif [[ -f "${REPORT_STEM}.qdstrm" ]]; then
    REPORT="${REPORT_STEM}.qdstrm"
else
    REPORT_TMP=$(ls -t /tmp/nsys-report-*.qdstrm 2>/dev/null | head -1 || true)
    [[ -z "$REPORT_TMP" ]] && { echo "ERROR: could not locate nsys report" >&2; exit 1; }
    REPORT="${REPORT_STEM}.qdstrm"
    cp "$REPORT_TMP" "$REPORT"
    echo "Report copied from $REPORT_TMP -> $REPORT"
fi

python3 - "$META_FILE" "$REPORT" <<'EOF'
import sys, json
meta_path, report = sys.argv[1], sys.argv[2]
with open(meta_path) as f:
    meta = json.load(f)
meta["report_file"] = report
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
EOF

if [[ "$REPORT" == *.qdstrm ]]; then
    echo "nsys importer not available on this host."
    echo "Copy $REPORT and $META_FILE to a machine with full Nsight Systems, then run:"
    echo "  python analysis/nsys_to_json.py --report <converted>.nsys-rep --meta $META_FILE --output results/${REPORT_STEM}.json"
else
    echo "Converting $REPORT to JSON..."
    python3 "$SCRIPT_DIR/analysis/nsys_to_json.py" \
        --report "$REPORT" \
        --meta   "$META_FILE" \
        --output "results/${REPORT_STEM}.json"
fi
