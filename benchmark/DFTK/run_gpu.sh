#!/usr/bin/env bash
RESULTS_DIR=results_gpu_large
mkdir -p $RESULTS_DIR
source ~/dftk-venv/bin/activate
julia --project=../cuda_project runDFTK.jl --gpu --ecut=30 --kgrid=1,1,1 --results-dir="$RESULTS_DIR" "$@"
python3 plot_timings.py "$RESULTS_DIR/timings.csv" -o "$RESULTS_DIR/$(basename "${1%.*}").pdf"
