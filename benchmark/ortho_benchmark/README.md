# Orthogonalization Benchmarks

Benchmarks all orthogonalization methods (`:mgs`, `:mgs2`, `:qr`, `:cholqr2`, `:cholqr3`) across a range of matrix sizes and saves timing/stability plots as PDFs.

Instantiate the project once before the first run:
```bash
julia --project=cpu_project  -e 'using Pkg; Pkg.instantiate()'
julia --project=cuda_project -e 'using Pkg; Pkg.instantiate()'
julia --project=rocm_project -e 'using Pkg; Pkg.instantiate()'
```

**CPU:**
```bash
julia --project=cpu_project bench_ortho.jl
```

**GPU (CUDA):**
```bash
julia --project=cuda_project bench_ortho.jl --cuda
```

**GPU (ROCm/AMD):**
```bash
julia --project=rocm_project bench_ortho.jl --rocm
```
