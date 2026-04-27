#!/usr/bin/env bash
# Run GPU profiling. Timings are extracted afterwards with nsys stats.
#
# Usage:
#   ./run_gpu_profile.sh [options] structure.extxyz
#
# Options:
#   --solver    jd|jd_sketched  (default: jd)
#   --ecut      FLOAT           (default: 30)
#   --kgrid     I,J,K           (default: 2,2,2)
#   --scf-maxiter INT           (default: 10)
#   --threads   INT             Julia threads (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVER="jd"
ECUT=30
KGRID="2,2,2"
SCF_MAXITER=10
THREADS=1
STRUCTURE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)      SOLVER="$2";      shift 2 ;;
        --ecut)        ECUT="$2";        shift 2 ;;
        --kgrid)       KGRID="$2";       shift 2 ;;
        --scf-maxiter) SCF_MAXITER="$2"; shift 2 ;;
        --threads)     THREADS="$2";     shift 2 ;;
        --*)           echo "Unknown option: $1" >&2; exit 1 ;;
        *)             STRUCTURE="$1";   shift ;;
    esac
done

[[ -z "$STRUCTURE" ]] && { echo "Usage: $0 [options] structure.extxyz" >&2; exit 1; }

SYSTEM="$(basename "${STRUCTURE%.*}")"
REPORT="${SYSTEM}_${SOLVER}_gpu.nsys-rep"

nsys launch \
    --force-overwrite true \
    --output "${REPORT%.nsys-rep}" \
    julia \
        --project="$SCRIPT_DIR" \
        --threads="$THREADS" \
        "$SCRIPT_DIR/run_gpu_profile.jl" \
        "$STRUCTURE" \
        --solver "$SOLVER" \
        --ecut "$ECUT" \
        --kgrid "$KGRID" \
        --scf-maxiter "$SCF_MAXITER"

echo ""
echo "Profile report: $REPORT"
echo "View timeline:  nsys-ui $REPORT"
echo "NVTX summary:   nsys stats --report nvtx_sum $REPORT"
