#!/bin/bash

export OMP_NUM_THREADS=4

julia --project=. sparse.jl
