#!/usr/bin/env bash
# Run CPU benchmark for a single structure.
#
# Usage:
#   ./run_cpu.sh [options] structure.extxyz
#
# Options are passed through to runDFTK.jl, e.g.:
#   --pseudo-set=mpc_azora   Use local mpc-azora UPF pseudopotentials
#   --ecut=N                 Plane-wave cutoff (default: 40)
#   --kgrid=N,N,N            k-point grid (default: 1,1,1)
#   --results-dir=DIR        Output directory (default: results_cpu_large_ecut)

RESULTS_DIR=results_cpu_large_ecut
source ~/dftk-venv/bin/activate
mkdir -p $RESULTS_DIR
julia --project=../cpu_project runDFTK.jl --results-dir="$RESULTS_DIR" --kgrid=1,1,1 --ecut=40 "$@"
python3 plot_timings.py "$RESULTS_DIR/timings.csv" -o "$RESULTS_DIR/$(basename "${1%.*}").pdf"
