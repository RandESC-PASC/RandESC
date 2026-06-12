#!/usr/bin/env bash
julia --project=../cuda_project runDFTK.jl --gpu "$@"
