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

nsys launch \
    julia \
        --project="$SCRIPT_DIR" \
        --threads="${SLURM_CPUS_PER_TASK:-4}" \
        "$SCRIPT_DIR/run_gpu_profile.jl" \
        "$STRUCTURE" \
        --solver "$SOLVER" \
        --ecut "$ECUT" \
        --kgrid "$KGRID" \
        --scf-maxiter "$SCF_MAXITER"

echo "Converting nsys report to JSON..."
python "$SCRIPT_DIR/analysis/nsys_to_json.py" \
    --report report1.nsys-rep \
    --meta   "${SYSTEM}_${SOLVER}_gpu_meta.json" \
    --output "results/${SYSTEM}_${SOLVER}_gpu.json"
