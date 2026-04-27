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
# View locally:      nsys-ui <report>.nsys-rep
# NVTX summary:      nsys stats --report nvtx_sum <report>.nsys-rep
#
# Submit with:
#   sbatch submit_slurm_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRUCTURE="../example/DFTK_solver/si.extxyz"
SOLVER="jd"
ECUT=30
KGRID="2,2,2"
SCF_MAXITER=10

# Uncomment and adapt for your cluster:
# module load julia/1.11 cuda/12

nsys launch \
    julia \
        --project="$SCRIPT_DIR" \
        --threads=1 \
        "$SCRIPT_DIR/run_gpu_profile.jl" \
        "$STRUCTURE" \
        --solver "$SOLVER" \
        --ecut "$ECUT" \
        --kgrid "$KGRID" \
        --scf-maxiter "$SCF_MAXITER"

echo "NVTX summary:"
nsys stats --report nvtx_sum report1.nsys-rep
