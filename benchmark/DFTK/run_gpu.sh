#!/usr/bin/env bash
julia --project=../cuda_project runDFTK.jl --gpu "$@"
python3 plot_timings.py results/timings.csv -o "results/$(basename "${1%.*}").pdf"
