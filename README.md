# RandESC
## Solve eigenvalue problems using randomized algorithms

Install the package (creating a Julia environment in a different folder with RandESC in it):
```
# enter Julia REPL console:
julia
julia> import Pkg
julia> Pkg.develop(path="/path/to/RandESC/)
```

Run tests:
```
julia --project=. -e 'using Pkg; Pkg.test()'
# or
julia --project=. test/runtests.jl
```
This has to be done from inside the source directory of this repository, otherwise the path must be specified in the option `--project=`

## Source Files

| File | Description |
|------|-------------|
| `src/RandESC.jl` | Module entry point: exports, includes, and shared utilities |
| `src/interface.jl` | Unified `randESCSolver()` entry point providing access to all solvers |
| `src/sketch.jl` | Randomized embedding/sketching operators: Gaussian, SRTT (FFT/DCT-based), sparsesign, sparsestack |
| `src/jd.jl` | Blocked Jacobi-Davidson with soft locking. |
| `src/jd_sketched.jl` | Blocked Jacobi-Davidson with sketched (Θ-norm) orthogonalization (Balabanov-Grigori RGS). |
| `src/randomization_utils.jl` | Shared utilities for randomized solvers (sketch application, RGS orthogonalization) |

## Examples

| Path | Description |
|------|-------------|
| `example/dense_hermitian.jl` | Basic usage: constructs a random Hermitian matrix and runs solvers |
| `example/DFTK_solver/runDFTK.jl` | Integration with [DFTK.jl](https://dftk.org): plugs RandESC solvers into a DFT self-consistent field calculation as a custom eigensolver |
