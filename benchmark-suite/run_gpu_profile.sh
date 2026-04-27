#!/usr/bin/env bash
# Run GPU profiling and convert the nsys report to a JSON timing file.
#
# Usage:
#   ./run_gpu_profile.sh [options] structure.extxyz
#
# Options:
#   --solver    jd|jd_sketched  (default: jd)
#   --ecut      FLOAT           (default: 30)
#   --kgrid     I,J,K           (default: 2,2,2)
#   --scf-maxiter INT           (default: 10)
#   --output    DIR             output directory for JSON (default: results)
#   --threads   INT             Julia threads (default: 1)

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
REPORT_STEM="${SYSTEM}_${SOLVER}_gpu"
META_FILE="${REPORT_STEM}_meta.json"
JSON_OUT="$OUTPUT/${REPORT_STEM}.json"

mkdir -p "$OUTPUT"

echo "========================================"
echo "GPU profiling: $SYSTEM / $SOLVER"
echo "========================================"

# nsys launch (unlike nsys profile) lets Julia exit naturally before writing
# the report, so metadata can be written after the @profile block.
nsys launch \
    --force-overwrite true \
    --output "${REPORT_STEM}" \
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
        --report "$REPORT" --meta "$META_FILE" --output "$JSON_OUT"
    echo "Done. JSON written to $JSON_OUT"
else
    echo "Report not found at $REPORT — nsys importer may be unavailable on this host."
    echo "Copy the .qdstrm file from /tmp and $META_FILE to a machine with full Nsight Systems."
fi
