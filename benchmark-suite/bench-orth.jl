# Benchmark: orthogonalization methods in RandESC vs DFTK Cholesky-based ortho
# A=I (standard eigenvalue problem), random input matrices, increasing n, fixed p.
# Measures wall time and output quality (||Q'Q - I||_F).

using LinearAlgebra
using Random
using BenchmarkTools
using Plots
using RandESC

BLAS.set_num_threads(8)
# Threads.nthreads() == 1 || @warn "Julia started with $(Threads.nthreads()) threads; run with `julia -t 1` for single-threaded benchmarks"

# ── DFTK Cholesky ortho (inlined from DFTK lobpcg_hyper_impl.jl) ─────────────

function _normest(M)
    maximum(abs, diag(M)) + norm(M - Diagonal(diag(M)))
end

function _safe_cholesky(O::AbstractMatrix{T}; nchol=0, α=100.0) where {T}
    nchol >= 5 && return nothing, nothing, 10000
    local R, invR
    try
        nchol += 1
        R = cholesky(O).U
        invR = inv(R)
        any(isnan, invR) && error("NaN in invR")
    catch
        O = O + α * eps(real(T)) * norm(O) * I
        α *= 10
        return _safe_cholesky(O; nchol, α)
    end
    R, invR, nchol
end

"""DFTK-style repeated Cholesky orthonormalization (in-place)."""
function dftk_ortho!(X::AbstractMatrix{T}; tol=2eps(real(T))) where {T}
    while true
        O = Hermitian(X' * X)
        R, invR, nchol = _safe_cholesky(O)
        if nchol > 10
            U, _, V = svd(X)
            X .= U * V'
            return X
        end
        rmul!(X, invR)
        norminvR = _normest(invR)
        condR    = _normest(R) * norminvR
        estimated_error = eps(real(T)) * condR^2
        nchol == 1 && estimated_error < tol && break
    end
    X
end

# ── Benchmark helpers ─────────────────────────────────────────────────────────

function orth_quality(Q)
    G = Q' * Q
    norm(G - I)
end

# Wrap RandESC methods to the same in-place signature
function bench_randesc!(V, method)
    m = size(V, 2)
    buf = similar(V, size(V, 1), m)
    RandESC._jd_ortho!(V, m, method, buf)
    return V
end

function bench_dftk!(V)
    dftk_ortho!(V)
    return V
end

# ── Experiment parameters ─────────────────────────────────────────────────────

ns = [1_000, 2_000, 5_000, 10_000, 20000, 50000]
ps = round.(Int, 0.01 .* ns)   # p = 5% of n
n_trials = 3

methods = [:mgs, :mgs2, :qr, :dftk]
labels  = Dict(:mgs => "MGS", :mgs2 => "MGS2", :qr => "Householder QR", :dftk => "DFTK Cholesky")
colors  = Dict(:mgs => :blue, :mgs2 => :orange, :qr => :green, :dftk => :red)

# Results: times[method][i] and quality[method][i] for ns[i]
times   = Dict(m => zeros(length(ns)) for m in methods)
quality = Dict(m => zeros(length(ns)) for m in methods)

@info "Benchmarking orthogonalization methods (p = 5% of n)"

for (i, n) in enumerate(ns)
    p = ps[i]
    @info "  n = $n, p = $p"
    Random.seed!(42)

    for method in methods
        t_total = 0.0
        q_total = 0.0
        for _ in 1:n_trials
            V0 = randn(n, p)
            V  = copy(V0)
            t = @elapsed begin
                if method === :dftk
                    bench_dftk!(V)
                else
                    bench_randesc!(V, method)
                end
            end
            t_total += t
            q_total += orth_quality(V[:, 1:p])
        end
        times[method][i]   = t_total / n_trials
        quality[method][i] = q_total / n_trials
    end
end

# ── Plots ─────────────────────────────────────────────────────────────────────

mkpath(joinpath(@__DIR__, "results"))

# Time plot
pt = plot(title="Orthogonalization: wall time vs n",
          xlabel="n (ambient dimension)",
          ylabel="time (s)",
          xscale=:log10, yscale=:log10,
          legend=:topleft, size=(800, 500))
for m in methods
    plot!(pt, ns, times[m]; label=labels[m], color=colors[m], marker=:circle, lw=2)
end
savefig(pt, joinpath(@__DIR__, "results", "orth_time.pdf"))

# Quality plot
pq = plot(title="Orthogonalization: output quality vs n",
          xlabel="n (ambient dimension)",
          ylabel="‖Q'Q − I‖_F",
          xscale=:log10, yscale=:log10,
          legend=:topleft, size=(800, 500))
for m in methods
    plot!(pq, ns, quality[m]; label=labels[m], color=colors[m], marker=:circle, lw=2)
end
savefig(pq, joinpath(@__DIR__, "results", "orth_quality.pdf"))

@info "Plots saved to benchmark-suite/results/ (PDF)"

# ── Summary table ─────────────────────────────────────────────────────────────

println("\n=== Time (s) ===")
print(rpad("n", 8)); print(rpad("p", 8))
for m in methods; print(rpad(labels[m], 18)); end
println()
for (i, n) in enumerate(ns)
    print(rpad(n, 8)); print(rpad(ps[i], 8))
    for m in methods; print(rpad(round(times[m][i]; sigdigits=3), 18)); end
    println()
end

println("\n=== Quality ‖Q'Q − I‖_F ===")
print(rpad("n", 8)); print(rpad("p", 8))
for m in methods; print(rpad(labels[m], 18)); end
println()
for (i, n) in enumerate(ns)
    print(rpad(n, 8)); print(rpad(ps[i], 8))
    for m in methods; print(rpad(round(quality[m][i]; sigdigits=3), 18)); end
    println()
end
