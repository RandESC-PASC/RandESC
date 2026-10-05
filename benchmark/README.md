# RandESC Benchmark Suite

This directory contains benchmarking tools for RandESC's eigensolvers,
organized into two parts:

- **`DFTK/`** – DFTK-based SCF benchmarks comparing LOBPCG, Jacobi-Davidson
  (`jd`), and sketched Jacobi-Davidson (`jd_sketched`) on real materials.
- **`ortho_benchmark/`** – Standalone orthogonalization micro-benchmarks
  (`:mgs`, `:mgs2`, `:qr`, `:cholqr2`, `:cholqr3`) on CPU, CUDA, and ROCm.

## Quick start

### 1. Install Julia dependencies

The DFTK benchmarks use separate project environments for CPU and GPU runs.

```bash
cd benchmark/cpu_project
julia --project=. -e 'using Pkg; Pkg.instantiate()'

cd ../cuda_project
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Run a single CPU benchmark

```bash
cd benchmark/DFTK
./run_cpu.sh ../mc3d_optimade_cpu/ClOTi_b01d85d1-aea7-4301-87e3-a694a578884f.cif.extxyz
```

This appends one row per algorithm to
`results_cpu_large_ecut/timings.csv` and produces a PDF plot.

### 3. Run a single GPU benchmark

```bash
cd benchmark/DFTK
./run_gpu.sh ../mc3d_optimade_gpu_large/some_supercell.extxyz
```

GPU profiling uses NVTX; the resulting `report*.nsys-rep` files can be opened
with `nsys-ui`.

## `runDFTK.jl` CLI options

All options are passed through by `run_cpu.sh` and `run_gpu.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu` | off | Run on the first CUDA device |
| `--ecut=N` | 45 (CPU), 30 (GPU) | Plane-wave cutoff in Hartree |
| `--kgrid=N,N,N` | `2,2,2` (CPU), `1,1,1` (GPU) | Monkhorst-Pack k-point grid |
| `--results-dir=DIR` | `results` | Output directory for `timings.csv` |
| `--pseudo-set=SET` | `default` | Pseudopotential set (see below) |

## Pseudopotential options

### `default` (CP2K GTH registry)

Uses DFTK's built-in registry family:

```julia
PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
```

This is the original behaviour and works for all elements covered by that
registry.

### `mpc_azora` (local UPF files)

Uses the mpc-azora-psp pseudopotentials converted to UPF and stored in
`benchmark/pseudos/mpc_azora_upf/`. These cover:

`H`, `Li`, `Be`, `B`, `C`, `N`, `O`, `F`, `Na`, `Mg`, `Al`, `Si`, `P`, `S`,
`Cl`, `K`, `Ca`, `Sc`, `Ti`, `V`, `Cr`, `Mn`, `Fe`, `Co`, `Ni`, `Cu`, `Zn`,
`Ga`, `Ge`, `As`, `Se`, `Br`.

Example:

```bash
./run_cpu.sh --pseudo-set=mpc_azora ../mc3d_optimade_cpu/ClOTi_b01d85d1-aea7-4301-87e3-a694a578884f.cif.extxyz
```

If a structure contains elements **not** in the mpc-azora set, the benchmark
prints a warning and skips that structure (no SCF is run).

### Regenerating the mpc-azora UPFs

The local UPFs were generated from `new_pseudos/SI/mpc_azora_psp/` with a
patched `hgh2qe.x` converter. To regenerate them:

```bash
cd benchmark
python3 convert_mpc_azora.py
```

This builds `benchmark/tools/hgh2qe_mpc.x` if needed, runs the conversion,
checks for Fortran `***` overflow markers, and validates every file with
`DFTK.load_psp`. See `pseudos/README.md` for details.

## Batch runs and SLURM

`submit.sh` is an example SLURM script that loops over
`../mc3d_optimade_gpu_large/*.extxyz`. Set the `PSEUDO_SET` variable near the
top to switch pseudopotential sets.

## Analysis

After producing `timings.csv`:

```bash
cd benchmark/DFTK
python3 summarize_timings.py results_cpu_large_ecut/timings.csv
python3 plot_timings.py results_cpu_large_ecut/timings.csv
```

## Structure sets

- `mc3d_optimade/` – 100 medium-sized structures (Z ≤ 36).
- `mc3d_optimade_cpu/` – 50 smaller structures for CPU benchmarking.
- `mc3d_optimade_gpu_large/` – 25 larger supercells for GPU benchmarking.

New structures can be fetched from the MC3D database via OPTIMADE using
`get_structures_optimade.py`.

## Orthogonalization benchmarks

See `ortho_benchmark/README.md` for instructions on running the standalone
orthogonalization micro-benchmarks.
