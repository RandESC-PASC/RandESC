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

# Record a sentinel file so we can find the nsys output by mtime afterwards.
# nsys may write the report to /tmp as a .qdstrm when the importer is unavailable.
SENTINEL=$(mktemp)

nsys profile \
    --capture-range=cudaProfilerApi \
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
        --scf-maxiter "$SCF_MAXITER" || true  # nsys exits non-zero after SIGTERM

rm -f "$SENTINEL"

# Locate the report: nsys writes <stem>.nsys-rep in the CWD when the importer
# is available, otherwise falls back to a .qdstrm in /tmp.
echo ""
if [[ -f "${REPORT_STEM}.nsys-rep" ]]; then
    REPORT="${REPORT_STEM}.nsys-rep"
    echo "Report: $REPORT"
elif [[ -f "${REPORT_STEM}.qdstrm" ]]; then
    REPORT="${REPORT_STEM}.qdstrm"
    echo "Report: $REPORT"
else
    # nsys wrote to /tmp — find the most recently created nsys qdstrm there
    REPORT_TMP=$(ls -t /tmp/nsys-report-*.qdstrm 2>/dev/null | head -1 || true)
    if [[ -z "$REPORT_TMP" ]]; then
        echo "ERROR: could not locate nsys report (checked CWD and /tmp)" >&2
        exit 1
    fi
    REPORT="${REPORT_STEM}.qdstrm"
    cp "$REPORT_TMP" "$REPORT"
    echo "Report copied from $REPORT_TMP -> $REPORT"
fi

# Patch report_file into the metadata JSON using python (avoids jq dependency)
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
    echo ""
    echo "nsys importer not available on this host."
    echo "Copy $REPORT and $META_FILE to a machine with full Nsight Systems, then run:"
    echo "  python analysis/nsys_to_json.py --report <converted>.nsys-rep --meta $META_FILE --output $JSON_OUT"
else
    echo "Converting $REPORT to JSON..."
    python3 "$SCRIPT_DIR/analysis/nsys_to_json.py" \
        --report "$REPORT" \
        --meta   "$META_FILE" \
        --output "$JSON_OUT"
    echo "Done. JSON written to $JSON_OUT"
fi
