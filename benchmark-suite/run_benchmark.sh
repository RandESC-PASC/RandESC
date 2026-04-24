#!/usr/bin/env bash
# Run benchmarks locally for one or more structures.
#
# Usage:
#   ./run_benchmark.sh [options] structure1.extxyz [structure2.extxyz ...]
#
# Options:
#   --output DIR          Output directory for JSON results (default: results)
#   --ecut FLOAT          Plane-wave cutoff in Ry (default: 30)
#   --kgrid I,J,K         k-point grid (default: 2,2,2)
#   --solvers S1,S2,...   Comma-separated solver list (default: lobpcg,jd,jd_sketched)
#   --julia-threads INT   Julia threads for k-point parallelism (default: 1)
#   --fftw-threads INT    FFTW threads (default: 1)
#   --blas-threads INT    BLAS threads (default: 1)
#
# Example:
#   ./run_benchmark.sh --julia-threads 2 --fftw-threads 4 --blas-threads 4 \
#       --ecut 40 --kgrid 3,3,3 ../example/DFTK_solver/h3s.extxyz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT="results"
ECUT=40
KGRID="1,1,1"
SOLVERS="lobpcg,jd,jd_sketched"
JULIA_THREADS=2
FFTW_THREADS=2
BLAS_THREADS=2
STRUCTURES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)        OUTPUT="$2";        shift 2 ;;
        --ecut)          ECUT="$2";          shift 2 ;;
        --kgrid)         KGRID="$2";         shift 2 ;;
        --solvers)       SOLVERS="$2";       shift 2 ;;
        --julia-threads) JULIA_THREADS="$2"; shift 2 ;;
        --fftw-threads)  FFTW_THREADS="$2";  shift 2 ;;
        --blas-threads)  BLAS_THREADS="$2";  shift 2 ;;
        --*)             echo "Unknown option: $1" >&2; exit 1 ;;
        *)               STRUCTURES+=("$1"); shift ;;
    esac
done

if [[ ${#STRUCTURES[@]} -eq 0 ]]; then
    echo "Error: no structure files specified." >&2
    echo "Usage: $0 [options] structure1.extxyz [structure2.extxyz ...]" >&2
    exit 1
fi

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

echo ""
echo "All benchmarks done. Results written to: $OUTPUT"
