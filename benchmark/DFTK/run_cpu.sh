#!/usr/bin/env bash
julia --project=../cpu_project runDFTK.jl "$@"
python3 plot_timings.py results/timings.csv -o "results/$(basename "${1%.*}").pdf"
