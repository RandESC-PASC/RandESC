# Randomized Embeddings & Eigensolvers (Julia)

Simple Julia code for sketching (randomized embeddings) and eigensolvers.

- `sketch.jl` — randomized embedding that returns a callable action `X ↦ S*X`
- `IRA.jl` — Implicitly Restarted Arnoldi (deterministic)
- `RIRA.jl` — Implicitly Restarted Arnoldi with **sketched** orthogonalization
- `LOBPCG.jl` — Block PCG for symmetric/Hermitian eigenproblems (no locking/deflation)

---

## Quick start

Place the files in your project and `include` what you need:

```julia
include("sketch.jl")
include("IRA.jl")
include("RIRA.jl")
include("LOBPCG.jl")
