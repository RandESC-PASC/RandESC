#!/usr/bin/env bash
# Run GPU profiling and immediately convert the nsys report to a JSON timing file.
#
# Usage:
#   ./run_gpu_profile.sh [options] structure.extxyz
#
# Options:
#   --solver    jd|jd_sketched  (default: jd)
#   --ecut      FLOAT           (default: 30)
#   --kgrid     I,J,K           (default: 2,2,2)
#   --scf-maxiter INT           SCF iterations to profile (default: 10)
#   --output    DIR             Output directory for JSON (default: results)
#   --threads   INT             Julia threads (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVER="jd"
ECUT=30
KGRID="2,2,2"
SCF_MAXITER=10
OUTPUT="results"
THREADS=1
STRUCTURE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)      SOLVER="$2";      shift 2 ;;
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
META_FILE="${SYSTEM}_${SOLVER}_gpu_meta.json"
REPORT="report1.nsys-rep"
JSON_OUT="$OUTPUT/${SYSTEM}_${SOLVER}_gpu.json"

mkdir -p "$OUTPUT"

echo "========================================"
echo "GPU profiling: $SYSTEM / $SOLVER"
echo "========================================"

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

echo ""
REPORT="$(python3 -c "import json; print(json.load(open('$META_FILE'))['report_file'])")"
echo "Converting $REPORT to JSON..."
python "$SCRIPT_DIR/analysis/nsys_to_json.py" \
    --report "$REPORT" \
    --meta   "$META_FILE" \
    --output "$JSON_OUT"

echo "Done. JSON written to $JSON_OUT"
