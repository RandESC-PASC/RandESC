#!/usr/bin/env bash
RESULTS_DIR=results_cpu_large_ecut
source ~/dftk-venv/bin/activate
mkdir -p $RESULTS_DIR
julia --project=../cpu_project runDFTK.jl --results-dir="$RESULTS_DIR" --kgrid=1,1,1 --ecut=40 "$@"
python3 plot_timings.py "$RESULTS_DIR/timings.csv" -o "$RESULTS_DIR/$(basename "${1%.*}").pdf"
