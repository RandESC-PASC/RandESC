#!/bin/bash
#SBATCH --job-name=gh200
#SBATCH --account=lp18
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=default

source ~/dftk-venv/bin/activate

OUTDIR=outputs_gpu_large_sys

mkdir -p $OUTDIR
for f in ../mc3d_optimade_gpu_large/*.extxyz; do
    srun bash run_gpu.sh $f &> $OUTDIR/$(basename $f .extxyz).out
done

