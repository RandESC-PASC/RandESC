#!/usr/bin/env bash
# Run GPU benchmark for a single structure.
#
# Usage:
#   ./run_gpu.sh [options] structure.extxyz
#
# Options are passed through to runDFTK.jl, e.g.:
#   --pseudo-set=mpc_azora   Use local mpc-azora UPF pseudopotentials
#   --ecut=N                 Plane-wave cutoff (default: 30)
#   --kgrid=N,N,N            k-point grid (default: 1,1,1)
#   --results-dir=DIR        Output directory (default: results_gpu_large)

RESULTS_DIR=results_gpu_large
mkdir -p $RESULTS_DIR
source ~/dftk-venv/bin/activate
julia --project=../cuda_project runDFTK.jl --gpu --ecut=30 --kgrid=1,1,1 --results-dir="$RESULTS_DIR" "$@"
python3 plot_timings.py "$RESULTS_DIR/timings.csv" -o "$RESULTS_DIR/$(basename "${1%.*}").pdf"
