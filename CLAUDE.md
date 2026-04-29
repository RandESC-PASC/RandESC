# RandESC

Julia library for solving Hermitian eigenvalue problems using randomized algorithms. Implements Jacobi-Davidson solvers with optional randomized (sketched) variants.

## Build & Test

```bash
# Run all tests (unit + integration)
julia --project=. -e 'using Pkg; Pkg.test("RandESC")'

# Run only unit tests (no external dependencies)
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["unit"])'

# Run only integration tests (requires DFTK)
julia --project=. -e 'using Pkg; Pkg.test("RandESC"; test_args=["integration"])'
```

Unit tests generate random symmetric/Hermitian matrices and verify solvers against `LinearAlgebra.eigen()`. The orthogonalization unit tests check all method variants directly.

## Project Structure

- `src/RandESC.jl` — module definition and exports
- `src/interface.jl` — unified `randESCSolver()` entry point with access to all solvers
- `src/jd.jl` — blocked Jacobi-Davidson with soft locking; recommended default solver
- `src/jd_sketched.jl` — blocked Jacobi-Davidson with sketched (Θ-norm) orthogonalization; mirrors `jd.jl` but uses sketch-based CGS instead of standard orthogonalization
- `src/orthogonalization.jl` — all orthogonalization routines (standard and sketched); main dispatch points are `_ortho!` and `_sketch_ortho!`
- `src/sketch.jl` — randomized embedding/sketching operators (`MatrixSketchOp`, `SRTTSketchOp`); sketch types: Gaussian, SRTT (FFT/DCT-based), sparsesign, sparsestack
- `src/timing.jl` — `@timing` macro (wraps `TimerOutputs`), `timer` global, optional NVTX hooks for GPU profiling
- `src/utils.jl` — device-agnostic helpers (`to_device`, `to_cpu`, `columnwise_norms`, `dct2`, sparse constructors)
- `test/runtests.jl` — test dispatcher (accepts `"unit"` / `"integration"` args)
- `test/unit/symmetric_hermitian_random_matrix.jl` — solver correctness tests for `jd` and `jd_sketched`
- `test/unit/test_orthogonalization.jl` — unit tests for all orthogonalization routines
- `test/integration/dftk_silicon_test.jl` — integration test against DFTK silicon problem
- `latex/presentation.tex` — theory for eigensolvers and randomization; detailed derivation of Jacobi-Davidson and its sketched variant

## Code Conventions

- **Functions**: `snake_case` (e.g., `rayleigh_ritz_standard`, `_sketch_ortho!`)
- **Internal functions**: prefixed with `_` (e.g., `_ortho!`, `_sketch_deflate!`)
- **Matrices**: uppercase single letters (`V`, `W`, `H`, `X`); sketched versions prefixed with `S` (`SV`, `SW`)
- **Vectors**: lowercase single letters (`v`, `w`, `r` for residual)
- **Eigenvalues**: `theta` (Ritz values), `lambda` (converged)
- Use `@views` to avoid array copies; prefer in-place ops (`mul!`, `ldiv!`)
- Operators can be `AbstractMatrix` or callable (`A * x` or `A(x)`)
- Return values use named tuples with descriptive fields (`λ`, `X`, `residual_norms`, `n_iter`, `converged`, `n_matvec`)
- Triple-quoted docstrings with `funcname(args) -> retval` header for all functions
- No trailing semicolons (Julia convention); semicolons only for statement separation on one line

## Key Concepts

- **Sketching**: Random dimensionality reduction `Θ: ℝⁿ → ℝˢ` (s << n) to make orthogonalization cheaper. Sketch types: Gaussian, SRTT (FFT/DCT-based), sparsesign, sparsestack.
- **Θ-orthonormality**: Vectors orthonormal in sketch space (`SV' * SV ≈ I`) rather than full space.
- **Soft locking**: Converged eigenpairs are "locked" (excluded from iteration) to reduce work.
- **Deflation**: Remove converged eigenvectors from the active search subspace.
- **Ritz pairs**: Approximate eigenpairs from projecting onto a subspace.
- **Orthogonalization methods** — standard: `:mgs` (single-pass MGS), `:mgs2` (double-pass), `:qr` (Householder QR); sketched: `:rcgs` (single-pass randomized CGS), `:rcgs2` (double-pass), `:rqr` (sketched QR, preferred on GPU).

## Preferred Solvers

- **`jd`** — recommended default. Blocked Jacobi-Davidson with soft locking, standard orthogonalization. Default `orth_method=:qr`.
- **`jd_sketched`** — recommended when orthogonalization is the bottleneck (large `n`). Same algorithm with Θ-norm sketch-based CGS orthogonalization. Default `orth_method=:rcgs`.

## Exported API

`jd`, `jd_sketched`, `randESCSolver`, `timer`, `reset_timer!`
