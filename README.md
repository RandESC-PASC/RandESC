# RandESC
## Solve eigenvalue problems using randomized algorithms

Install the package (creating a Julia environment in a different folder with RandESC in it):
```
# enter Julia REPL console:
julia
julia> import Pkg
julia> Pkg.develop(path="/path/to/RandESC/)
```

Run all tests (unit + integration):
```bash
julia --project=. -e 'using Pkg; Pkg.test("RandESC")'
```

Run only unit tests (no external dependencies):
```bash
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["unit"])'
```

Run only integration tests (requires DFTK; Silicon SCF calculation):
```bash
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["integration"])'
```

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
| `benchmark/DFTK/runDFTK.jl` | Integration with [DFTK.jl](https://dftk.org): plugs RandESC solvers into a DFT self-consistent field calculation as a custom eigensolver |

## DFTK benchmark setup

The DFTK benchmark driver `benchmark/DFTK/runDFTK.jl` uses `DFTK.timing_sync`, a hook that is not present in stock DFTK. To run the benchmark you must apply the included patch to a DFTK v0.7.25 source tree and pin `DftFunctionals` to v0.3.2.

1. Check out DFTK v0.7.25:

```bash
git clone https://github.com/JuliaMolSim/DFTK.jl.git
cd DFTK.jl
git checkout v0.7.25
```

2. Pin `DftFunctionals` to v0.3.2 in the benchmark environment:

```julia
using Pkg
Pkg.pin(PackageSpec(name="DftFunctionals", version="0.3.2"))
```

3. Apply the timing-sync patch:

```bash
cd DFTK.jl
patch -p1 < /path/to/RandESC/benchmark/DFTK/dftk_timing_sync_v0.7.25.patch
```

If line numbers differ and the patch does not apply cleanly, apply the same changes manually to `src/common/timer.jl`.

4. Use the patched DFTK in your benchmark project:

```julia
using Pkg
Pkg.develop(path="/path/to/DFTK.jl")
```

Then run `benchmark/DFTK/runDFTK.jl` as usual.
