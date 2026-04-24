#!/usr/bin/env bash
#SBATCH --job-name=randesc-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Submit with:
#   sbatch submit_slurm.sh
#
# Adjust the SBATCH directives and the variables below for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────

STRUCTURES=(
    "../example/DFTK_solver/h3s.extxyz"
    "../example/DFTK_solver/h3s_supercell.extxyz"
)

OUTPUT="results/slurm-${SLURM_JOB_ID:-local}"
ECUT=30
KGRID="2,2,2"
SOLVERS="lobpcg,jd,jd_sketched"

# Per the DFTK parallelization guide:
#   - Julia threads parallelize over k-points (keep low, e.g. 2)
#   - FFTW threads handle FFTs which dominate runtime (80–90%)
#   - BLAS threads handle dense linear algebra
# Recommended split for 8 CPUs: 2 Julia, 4 FFTW, 4 BLAS
NCPUS="${SLURM_CPUS_PER_TASK:-8}"
JULIA_THREADS=2
FFTW_THREADS=$(( NCPUS / 2 ))
BLAS_THREADS=$(( NCPUS / 2 ))

# ── Module / environment setup ────────────────────────────────────────────────
# Uncomment and adapt for your cluster's module system:
# module load julia/1.11
# module load gcc/12

# ── Run ───────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT"

for STRUCTURE in "${STRUCTURES[@]}"; do
    echo "========================================"
    echo "Structure: $STRUCTURE"
    echo "========================================"
    julia \
        --project="$SCRIPT_DIR" \
        --threads="$JULIA_THREADS" \
        "$SCRIPT_DIR/run_benchmark.jl" \
        "$STRUCTURE" \
        --output "$OUTPUT" \
        --ecut "$ECUT" \
        --kgrid "$KGRID" \
        --solvers "$SOLVERS" \
        --fftw-threads "$FFTW_THREADS" \
        --blas-threads "$BLAS_THREADS"
done

echo "Done. Results in: $OUTPUT"
