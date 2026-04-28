#!/usr/bin/env bash
# Run GPU profiling for jd and jd_sketched, parse NVTX timings to JSON.
#
# Usage:
#   ./run_gpu_profile.sh [options] structure.extxyz
#
# Options:
#   --solvers   COMMA,LIST      (default: jd,jd_sketched)
#   --ecut      FLOAT           (default: 30)
#   --kgrid     I,J,K           (default: 2,2,2)
#   --scf-maxiter INT           (default: 10)
#   --output    DIR             output directory for JSON (default: results)
#   --threads   INT             Julia threads (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVERS="jd"
ECUT=30
KGRID="2,2,2"
SCF_MAXITER=10
OUTPUT="results"
THREADS=1
STRUCTURE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solvers)     SOLVERS="$2";     shift 2 ;;
        --ecut)        ECUT="$2";        shift 2 ;;
        --kgrid)       KGRID="$2";       shift 2 ;;
        --scf-maxiter) SCF_MAXITER="$2"; shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --threads)     THREADS="$2";     shift 2 ;;
        --*)           echo "Unknown option: $1" >&2; exit 1 ;;
        *)             STRUCTURE="$1";   shift ;;
    esac
done

[[ -z "$STRUCTURE" ]] && { echo "Usage: $0 [options] structure.extxyz" >&2; exit 1; }

SYSTEM="$(basename "${STRUCTURE%.*}")"
mkdir -p "$OUTPUT"

IFS=',' read -ra SOLVER_LIST <<< "$SOLVERS"
for SOLVER in "${SOLVER_LIST[@]}"; do
    echo "========================================"
    echo "Profiling: $SYSTEM / $SOLVER"
    echo "========================================"

    # Note which reports exist before this run
    BEFORE=$(ls report*.nsys-rep 2>/dev/null | sort || true)

    nsys launch \
        julia \
            --project="$SCRIPT_DIR" \
            --threads="$THREADS" \
            "$SCRIPT_DIR/run_gpu_profile.jl" \
            "$STRUCTURE" \
            --solver "$SOLVER" \
            --ecut "$ECUT" \
            --kgrid "$KGRID" \
            --scf-maxiter "$SCF_MAXITER"

    # Find the report that appeared during this run
    REPORT=$(comm -13 <(echo "$BEFORE") <(ls report*.nsys-rep 2>/dev/null | sort) | head -1)
    if [[ -z "$REPORT" ]]; then
        REPORT=$(ls -t report*.nsys-rep 2>/dev/null | head -1)
    fi

    echo ""
    echo "Report: $REPORT"
    python3 "$SCRIPT_DIR/analysis/parse_nvtx.py" \
        --report "$REPORT" --solver "$SOLVER" --system "$SYSTEM" \
        --output "$OUTPUT/${SYSTEM}_${SOLVER}_gpu.json"
    echo ""
done
