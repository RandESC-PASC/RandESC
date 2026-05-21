# Orthogonalization Benchmarks

Benchmarks all orthogonalization methods (`:mgs`, `:mgs2`, `:qr`, `:cholqr2`, `:cholqr3`) across a range of matrix sizes and saves timing/stability plots as PDFs.

**CPU:**
```bash
julia --project=cpu_project bench_ortho.jl
```

**GPU (CUDA):**
```bash
julia --project=cuda_project bench_ortho.jl --cuda
```
