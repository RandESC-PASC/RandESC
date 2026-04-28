# RandESC

Julia library for solving Hermitian eigenvalue problems using randomized algorithms. Implements IRA, LOBPCG, and Jacobi-Davidson solvers, each with optional randomized (sketched) variants.

## Build & Test

```bash
# Run all tests (unit + integration)
julia --project=. -e 'using Pkg; Pkg.test("RandESC")'

# Run only unit tests (no external dependencies)
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["unit"])'

# Run only integration tests (requires DFTK)
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["integration"])'
```

Tests generate random symmetric matrices (n=50, k=7) and verify all solvers against `LinearAlgebra.eigen()`.

## Project Structure

- `src/RandESC.jl` — module definition, exports, utilities
- `src/interface.jl` — unified `randESCSolver()` entry point. provide a unified interface with acess to all solvers implemented in this package
- `src/sketch.jl` — randomized embedding/sketching operators (Gaussian, SRTT, sparsesign)
- `src/jd.jl` — blocked Jacobi-Davidson with soft locking; recommended default solver.
- `src/jd_sketched.jl` — blocked Jacobi-Davidson with sketched (Θ-norm) orthogonalization; mirrors `jd.jl` but uses RGS instead of MGS.
- `src/randomization_utils.jl` utils for randomization, most useful for jacobi davidson flavours
- `latex/presentation.tex` a tex file containing theory for eigensolvers and randomization. the theory of jacobi davidson and randomized jacobi davidson ist explained in detail in here
- `test/tests.jl` — test suite using `@testset`

## Code Conventions

- **Functions**: `snake_case` (e.g., `rayleigh_ritz_standard`, `arnoldi_sketch`)
- **Matrices**: uppercase single letters (`V`, `W`, `H`, `X`); sketched versions prefixed with `S` (`SV`, `SW`)
- **Vectors**: lowercase single letters (`v`, `w`, `r` for residual)
- **Eigenvalues**: `theta` (Ritz values), `lambda` (converged)
- **Algorithm variants**: suffixes like `_sketch`, `_lock`, `_softlock`, `_rand`
- Use `@views` to avoid array copies; prefer in-place ops (`mul!`, `ldiv!`)
- Operators can be `AbstractMatrix` or callable (`A * x` or `A(x)`)
- Return values use named tuples with descriptive fields (`λ`, `X`, `residual_norms`, `n_iter`, `converged`, `n_matvec`)
- Triple-quoted docstrings for public functions with argument descriptions
- No trailing semicolons (Julia convention); semicolons only used for statement separation on a single line

## Key Concepts

- **Sketching**: Random dimensionality reduction `Θ: ℝⁿ → ℝˢ` (s << n) to make orthogonalization cheaper. Sketch types: Gaussian, SRTT (FFT/DCT-based), sparsesign, sparsestack.
- **Θ-orthonormality**: Vectors orthonormal in sketch space (`SV' * SV ≈ I`) rather than full space.
- **Soft locking**: Converged eigenpairs are "locked" (excluded from iteration) to reduce work.
- **Deflation**: Remove converged eigenvectors from the active search subspace.
- **Ritz pairs**: Approximate eigenpairs from projecting onto a subspace.
- **Orthogonalization methods**: MGS, MGS2/DGKS, CGS, CGS2, RGS (sketch-based).

## Preferred Solvers

- **`jd`** — recommended default for most problems. Blocked Jacobi-Davidson with soft locking and standard MGS orthogonalization.
- **`jd_sketched`** — recommended when orthogonalization is the bottleneck (large `n`). Same algorithm but with sketched (Θ-norm) RGS orthogonalization.

## Exported API

`jd`, `jd_sketched`, `randESCSolver`
