using RandESC
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using CairoMakie

### Benchmark: jd vs jd_sketched on a sparse Hermitian operator.
###
### For each fixed k in K_VALUES we sweep an increasing matrix size n and record
### the wall time of both solvers (median over N_REPS runs). The result is a
### log-log timing-vs-n plot with one panel per k.
###
### Run with (CPU):
###   julia --project=benchmark/sparse benchmark/sparse/sparse.jl
###
### The first run instantiates the environment:
###   julia --project=benchmark/sparse -e 'using Pkg; Pkg.instantiate()'
###
### GPU is optional: CUDA/AMDGPU are NOT project dependencies, so the CPU env stays
### light. To run on GPU, add the driver package to this env on the GPU machine, then
### pass the flag:
###   julia --project=benchmark/sparse -e 'using Pkg; Pkg.add("CUDA")'   # or "AMDGPU"
###   julia --project=benchmark/sparse benchmark/sparse/sparse.jl --cuda  # or --rocm

# ── Configuration ─────────────────────────────────────────────────────────────
const K_VALUES = [100]                                  # fixed eigenpair counts
const N_VALUES = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
const N_REPS   = 3        # repetitions per (solver, k, n); mean ± std is reported
const TOL      = 1e-6     # residual convergence threshold
const MAXIT    = 500      # max JD iterations
const OP_KIND  = :logfun   # :model_hamiltonian | :logfun | :laplacian2d | :laplacian3d | :random_spd
const N_MIN_RATIO = 6     # skip a point unless n >= N_MIN_RATIO * k (keeps sketch s < n)

Random.seed!(1234)

# ── Device selection (CPU / CUDA / ROCm) ────────────────────────────────────────
# The solvers are architecture-agnostic: they inherit the device from v0 (dense)
# and operate on A through matvecs. We build the sparse operator on the host, then
# move it (and the dense arrays) to the requested device.
#
# CUDA/AMDGPU are loaded lazily and are NOT listed as project dependencies, so the
# CPU environment installs without any GPU packages. To run on GPU, `Pkg.add` the
# matching driver into this env first (see the header), then pass --cuda / --rocm.
const USE_CUDA = "--cuda" in ARGS
const USE_ROCM = "--rocm" in ARGS

# Load a GPU backend that is not a fixed dependency; give an actionable error if it
# has not been added to the active environment.
function require_backend(pkg)
    try
        @eval using $(Symbol(pkg))
    catch err
        error("`--$(lowercase(pkg))` was requested but $pkg is not installed in this " *
              "environment. Add it on the GPU machine with:\n" *
              "    julia --project=benchmark/sparse -e 'using Pkg; Pkg.add(\"$pkg\")'")
    end
end

if USE_CUDA
    require_backend("CUDA")
    to_device(x::AbstractArray) = CUDA.CuArray(x)
    to_device_sparse(A) = CUDA.CUSPARSE.CuSparseMatrixCSR(A)   # CSR preferred for SpMM
    gpu_sync() = CUDA.synchronize()
    println("Running on CUDA GPU")
elseif USE_ROCM
    require_backend("AMDGPU")
    to_device(x::AbstractArray) = AMDGPU.ROCArray(x)
    to_device_sparse(A) = AMDGPU.rocSPARSE.ROCSparseMatrixCSR(A)
    gpu_sync() = AMDGPU.synchronize()
    println("Running on ROCm GPU")
else
    to_device(x::AbstractArray) = x
    to_device_sparse(A) = A
    gpu_sync() = nothing
    println("Running on CPU")
end

const SUFFIX = USE_CUDA ? "_cuda" : USE_ROCM ? "_rocm" : ""

# ── Sparse operators ────────────────────────────────────────────────────────────
# All builders return (A, n_actual): a sparse SPD matrix and the realized size,
# which may differ from the requested n when the geometry forces rounding.

"""
    model_hamiltonian(n; nnz_per_row=10, gap=1.0, offdiag=0.05) -> (A, n)

Synthetic sparse Hermitian "model Hamiltonian" with a planted, well-separated low
spectrum. The diagonal is a ramp d[i] = i·gap (target eigenvalues spaced by `gap`);
a small symmetric sparse off-diagonal coupling (entry magnitude `offdiag` << `gap`,
~`nnz_per_row` per row) couples the modes while keeping the matrix strictly
diagonally dominant — hence SPD with simple, well-separated low eigenvalues. JD
converges quickly and reliably for any k and n, so wall-time differences reflect
solver cost (orthogonalization, which dominates as n grows) rather than stalls.
"""
function model_hamiltonian(n; nnz_per_row=10, gap=1.0, offdiag=0.05)
    p = min(1.0, nnz_per_row / n)
    B = sprandn(n, n, p) .* offdiag
    A = B + B'                              # symmetric off-diagonal coupling
    A = A - spdiagm(0 => diag(A))           # drop the random diagonal contribution
    A = A + spdiagm(0 => gap .* (1:n))      # plant well-separated spectrum
    return A, n
end

"""
    laplacian_2d(n) -> (A, n_actual)

5-point finite-difference Laplacian on a √m × √m grid (Dirichlet BC), returned as
a sparse SPD matrix with ~5 nonzeros per row. n is rounded to the nearest perfect
square. A mild anisotropy (1.1 on the y-direction) lifts the symmetry-induced
eigenvalue degeneracies so the low spectrum is simple and well separated, giving
fast and reliable Jacobi-Davidson convergence — a clean stress test for
orthogonalization cost.
"""
function laplacian_2d(n)
    m = max(2, round(Int, sqrt(n)))
    T = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
    Im = sparse(1.0I, m, m)
    A = kron(T, Im) + 1.1 * kron(Im, T)          # anisotropy breaks degeneracies
    return A, m * m
end

"""
    logfun(n; offdiag=0.2) -> (A, n)

Sparse symmetric tridiagonal operator with a logarithmic diagonal ramp
d[i] = log(100 + i - 1) (offset by 100 to stay positive and well away from zero)
plus a small random off-diagonal, each entry uniform in (-offdiag, offdiag). The
diagonal dominates the off-diagonal, so the matrix is SPD; the log ramp gives a
slowly-growing, bounded spectrum (κ ≈ log(100+n)/log(100)).
"""
function logfun(n; offdiag=0.2)
    d = log.(100 .+ (0:n-1))                       # log-spaced diagonal, length n
    e = offdiag .* (2 .* rand(n - 1) .- 1)         # off-diagonal in (-offdiag, offdiag)
    A = spdiagm(-1 => e, 0 => d, 1 => e)           # symmetric tridiagonal (sparse)
    return A, n
end

"""
    laplacian_3d(n) -> (A, n_actual)

7-point finite-difference Laplacian on a m × m × m grid. n is rounded so that
m = round(n^(1/3)). Mild anisotropy (1.1, 1.2 weights) lifts degeneracies.
Sparser per-row footprint relative to n than the 2D case.
"""
function laplacian_3d(n)
    m = max(2, round(Int, cbrt(n)))
    T = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
    Im = sparse(1.0I, m, m)
    A = kron(kron(T, Im), Im) + 1.1 * kron(kron(Im, T), Im) + 1.2 * kron(kron(Im, Im), T)
    return A, m * m * m
end

"""
    random_spd(n; nnz_per_row=10) -> (A, n)

Random symmetric sparse matrix made strictly diagonally dominant (hence SPD).
Generic operator with a tunable nonzero density; eigenvalues are less separated
than the Laplacians, so convergence is slower.
"""
function random_spd(n; nnz_per_row=10)
    p = min(1.0, nnz_per_row / n)
    A = sprandn(n, n, p)
    A = A + A'                                   # symmetric
    d = vec(sum(abs, A; dims=2)) .+ 1.0          # diagonal dominance => SPD
    A = A + spdiagm(0 => d)
    return A, n
end

function make_operator(kind, n)
    kind === :model_hamiltonian && return model_hamiltonian(n)
    kind === :logfun && return logfun(n)
    kind === :laplacian2d && return laplacian_2d(n)
    kind === :laplacian3d && return laplacian_3d(n)
    kind === :random_spd  && return random_spd(n)
    error("unknown operator kind: $kind")
end

# ── Timing ──────────────────────────────────────────────────────────────────────

"""
    bench(solver, A, M, k, v0s) -> (mean_seconds, std_seconds)

Time `solver(A, v0; k, tol, maxit, M)` once per initial guess in `v0s` and return
the mean wall time and its standard deviation (for error bars). The same `v0s` (and
the same preconditioner `M`) are passed to every solver so the comparison is on
identical starting subspaces. GPU work is synchronized so timings are accurate.
"""
function bench(solver, A, M, k, v0s)
    reps = length(v0s)
    times = Vector{Float64}(undef, reps)
    for r in 1:reps
        v0 = copy(v0s[r])         # copy outside timing; guard against in-place mutation
        gpu_sync()
        GC.gc()
        times[r] = @elapsed begin
            solver(A, v0; k=k, tol=TOL, maxit=MAXIT, M=M)
            gpu_sync()
        end
    end
    return mean(times), (reps > 1 ? std(times) : 0.0)
end

# Build the operator, the Jacobi preconditioner and the per-rep initial guesses for
# one (k, n) point, transferring everything to the active device. The diagonal is
# read from the host matrix to avoid a sparse-diag call on the GPU.
function prepare(n_req, k, reps)
    A_cpu, n = make_operator(OP_KIND, n_req)
    A = to_device_sparse(A_cpu)
    M = Diagonal(to_device(diag(A_cpu)))   # Jacobi precond (effective for diag-dominant A)
    T = eltype(A_cpu)
    v0s = [to_device(RandESC.random_matrix(T, n, k, nothing)) for _ in 1:reps]
    return A, M, n, v0s
end

# Warm up both solvers so JIT compilation is excluded from the measurements.
function warmup()
    A, M, _, v0s = prepare(400, 5, 1)
    jd(A, v0s[1]; k=5, tol=TOL, maxit=MAXIT, M=M)
    jd_sketched(A, v0s[1]; k=5, tol=TOL, maxit=MAXIT, M=M)
    gpu_sync()
    return nothing
end

# ── Run benchmark ─────────────────────────────────────────────────────────────
println("Operator: $OP_KIND")
println("k_values = $K_VALUES")
println("n_values = $N_VALUES")
println("Warming up (compiling solvers)...")
warmup()

# results[k] = (ns, mean_jd, std_jd, mean_sketch, std_sketch); only run n stored.
results = Dict{Int,NTuple{5,Vector{Float64}}}()

for k in K_VALUES
    ns       = Float64[]
    m_jd     = Float64[]; s_jd     = Float64[]
    m_sketch = Float64[]; s_sketch = Float64[]
    println("\n=== k = $k ===")
    for n_req in N_VALUES
        n_req < N_MIN_RATIO * k && continue           # keep sketch dim s < n
        # Same operator, preconditioner and initial guesses for both solvers.
        A, M, n, v0s = prepare(n_req, k, N_REPS)
        n < N_MIN_RATIO * k && continue
        @printf("  n=%-8d (requested %d) ... ", n, n_req)
        mj, sj = bench(jd, A, M, k, v0s)
        ms, ss = bench(jd_sketched, A, M, k, v0s)
        push!(ns, n)
        push!(m_jd, mj); push!(s_jd, sj)
        push!(m_sketch, ms); push!(s_sketch, ss)
        @printf("jd=%.3f±%.3fs  jd_sketched=%.3f±%.3fs  (speedup %.2fx)\n",
                mj, sj, ms, ss, mj / ms)
    end
    results[k] = (ns, m_jd, s_jd, m_sketch, s_sketch)
end

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = Makie.wong_colors()
fig = Figure(size=(450 * length(K_VALUES), 450))

# Plot mean wall time with ± std error bars. On a log y-axis the lower whisker
# is capped just above zero so mean−std never goes non-positive.
function plot_series!(ax, ns, m, s, color, label)
    lo = min.(s, 0.999 .* m)
    lines!(ax, ns, m; color=color, linewidth=2, label=label)
    scatter!(ax, ns, m; color=color, markersize=9)
    errorbars!(ax, ns, m, lo, s; color=color, whiskerwidth=10)
end

for (i, k) in enumerate(K_VALUES)
    ns, m_jd, s_jd, m_sketch, s_sketch = results[k]
    ax = Axis(fig[1, i];
        xlabel = "n",
        ylabel = i == 1 ? "Mean wall time [s]  (± std)" : "",
        title  = "k = $k",
        xscale = log10,
        yscale = log10,
    )
    isempty(ns) && continue
    plot_series!(ax, ns, m_jd, s_jd, colors[1], "jd")
    plot_series!(ax, ns, m_sketch, s_sketch, colors[2], "jd_sketched")
    axislegend(ax; position=:lt)
end

Label(fig[0, :], "jd vs jd_sketched on sparse $OP_KIND operator"; fontsize=18, font=:bold)

outfile = joinpath(@__DIR__, "sparse_timing_vs_n$(SUFFIX).benchmark.pdf")
save(outfile, fig)
println("\nSaved $outfile")
