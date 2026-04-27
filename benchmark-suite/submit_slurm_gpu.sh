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
# Produces <system>_<solver>_gpu.nsys-rep in the working directory.
# View locally with: nsys-ui <report>.nsys-rep
# Stats with:        nsys stats --report nvtx_sum <report>.nsys-rep
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

nsys launch \
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
        --scf-maxiter "$SCF_MAXITER"

REPORT="${REPORT_STEM}.nsys-rep"
if [[ -f "$REPORT" ]]; then
    python3 - "$META_FILE" "$REPORT" <<'EOF'
import sys, json
meta_path, report = sys.argv[1], sys.argv[2]
with open(meta_path) as f: meta = json.load(f)
meta["report_file"] = report
with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
EOF
    echo "Converting $REPORT to JSON..."
    python3 "$SCRIPT_DIR/analysis/nsys_to_json.py" \
        --report "$REPORT" --meta "$META_FILE" --output "results/${REPORT_STEM}.json"
else
    echo "Report not found at $REPORT — nsys importer may be unavailable on this host."
    echo "Copy the .qdstrm file from /tmp and $META_FILE to a machine with full Nsight Systems."
fi
