# RandESC Examples

## Setup

The examples use a shared Julia environment in this directory. Install all dependencies once from `example/`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Examples

### `dense_hermitian.jl`

Solves a small random dense Hermitian eigenvalue problem with both `jd` and `jd_sketched`, and compares the results against Julia's built-in `LinearAlgebra.eigen`.

```bash
julia --project=. dense_hermitian.jl
```

### `dftk_silicon.jl`

Runs a DFT self-consistent field calculation for a Silicon unit cell using [DFTK.jl](https://dftk.org), with RandESC as the eigensolver. Compares the standard and randomized block Jacobi-Davidson solvers.

```bash
julia --project=. dftk_silicon.jl
```

### `cuda_profiling.jl`

Creates a profile of a RandESC solver of choice on NVIDIA GPUs. More detailed instructions are
available in the file itself.

```bash
nsys launch julia cuda_profiling.jl
```
