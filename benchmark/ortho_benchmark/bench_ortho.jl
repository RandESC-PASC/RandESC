using RandESC
using LinearAlgebra
using Statistics
using CairoMakie

const USE_CUDA = "--cuda" in ARGS

if USE_CUDA
    using CUDA
    to_array(x) = CuArray(x)
    cpu_array(x) = Array(x)
    cuda_sync() = CUDA.synchronize()
    println("Running on CUDA GPU")
else
    to_array(x) = x
    cpu_array(x) = x
    cuda_sync() = nothing
    println("Running on CPU")
end

const METHODS = [:mgs, :mgs2, :qr, :cholqr2, :cholqr3]
const N_VALUES = [500, 1000, 2000, 5000, 10000, 20000, 50000]
const N_REPS   = 5   # repetitions per (method, n); take median
const SUFFIX   = USE_CUDA ? "_cuda" : ""

function bench_method(method::Symbol, n::Int, k::Int)
    times = Vector{Float64}(undef, N_REPS)
    orth_err = 0.0

    for r in 1:N_REPS
        V   = to_array(randn(n, k))
        buf = similar(V)
        cuda_sync()
        t = @elapsed begin
            RandESC._ortho!(V, k, method, buf)
            cuda_sync()
        end
        times[r] = t
        if r == N_REPS
            # orthogonality error on a fresh random matrix
            V2 = to_array(randn(n, k))
            RandESC._ortho!(V2, k, method, similar(V2))
            A = cpu_array(V2)
            orth_err = norm(A' * A - I(k))
        end
    end

    return median(times), orth_err
end

println("Benchmarking orthogonalization methods...")
println("n_values = $N_VALUES")
println("k = round(Int, 0.01 * n)")
println()

# results[method] = (times_vec, errs_vec)
results = Dict(m => (Float64[], Float64[]) for m in METHODS)

for n in N_VALUES
    k = max(1, round(Int, 0.01 * n))
    print("n=$n, k=$k ... ")
    for method in METHODS
        t, e = bench_method(method, n, k)
        push!(results[method][1], t)
        push!(results[method][2], e)
        print("$method=$(round(t*1e3, digits=1))ms ")
    end
    println()
end

# ── Plotting ────────────────────────────────────────────────────────────────

colors = Makie.wong_colors()
method_color = Dict(m => colors[i] for (i, m) in enumerate(METHODS))

# Wall-time plot
fig_time = Figure(size=(700, 450))
ax_time  = Axis(fig_time[1, 1];
    xlabel = "n  (k = 0.01·n)",
    ylabel = "Median wall time [s]",
    title  = "Orthogonalization wall time vs matrix size",
    xscale = log10,
    yscale = log10,
    xticks = (N_VALUES, string.(N_VALUES)),
    xticklabelrotation = π/4,
)

for method in METHODS
    times = results[method][1]
    lines!(ax_time, N_VALUES, times;
        label  = string(method),
        color  = method_color[method],
        linewidth = 2,
    )
    scatter!(ax_time, N_VALUES, times;
        color  = method_color[method],
        markersize = 8,
    )
end

Legend(fig_time[1, 2], ax_time, "Method")
save("time_vs_n$(SUFFIX).benchmark.pdf", fig_time)
println("\nSaved time_vs_n$(SUFFIX).benchmark.pdf")

# Stability plot
fig_stab = Figure(size=(700, 450))
ax_stab  = Axis(fig_stab[1, 1];
    xlabel = "n  (k = 0.01·n)",
    ylabel = "‖V'V − I‖_F",
    title  = "Orthogonalization stability vs matrix size",
    xscale = log10,
    yscale = log10,
    xticks = (N_VALUES, string.(N_VALUES)),
    xticklabelrotation = π/4,
)

for method in METHODS
    errs = results[method][2]
    lines!(ax_stab, N_VALUES, errs;
        label  = string(method),
        color  = method_color[method],
        linewidth = 2,
    )
    scatter!(ax_stab, N_VALUES, errs;
        color  = method_color[method],
        markersize = 8,
    )
end

Legend(fig_stab[1, 2], ax_stab, "Method")
save("stability_vs_n$(SUFFIX).benchmark.pdf", fig_stab)
println("Saved stability_vs_n$(SUFFIX).benchmark.pdf")

# ── Stability vs condition number (fixed n, k) ───────────────────────────────

const N_COND = 10_000
const K_COND = 100
# condition numbers from 1 to 1e15
const KAPPA_VALUES = 10 .^ (0:0.5:18)

"""
Build a random n×k matrix with prescribed condition number κ.
V = U * diag(s) * W' where U (n×k) and W (k×k) are random orthogonal matrices
and s spans [1, κ] geometrically. The Gram matrix V'V = W*diag(s²)*W' has
condition number κ², so the columns are genuinely nearly linearly dependent.
"""
function rand_matrix_with_condition(n, k, κ)
    U = Matrix(qr(randn(n, k)).Q)   # n×k orthogonal
    W = Matrix(qr(randn(k, k)).Q)   # k×k orthogonal
    s = exp.(range(0, log(κ), length=k))
    return to_array((U .* s') * W')           # U * diag(s) * W'
end

println("\nBenchmarking stability vs condition number (n=$N_COND, k=$K_COND)...")

cond_results = Dict(m => Float64[] for m in METHODS)

for κ in KAPPA_VALUES
    A = rand_matrix_with_condition(N_COND, K_COND, κ)
    for method in METHODS
        V   = copy(A)
        buf = similar(V)
        RandESC._ortho!(V, K_COND, method, buf)
        Vc = cpu_array(V)
        err = norm(Vc' * Vc - I(K_COND))
        push!(cond_results[method], err)
    end
    println("  κ=$(round(κ, sigdigits=2)): done")
end

fig_cond = Figure(size=(700, 450))
ax_cond  = Axis(fig_cond[1, 1];
    xlabel = "Condition number κ",
    ylabel = "‖V'V − I‖_F",
    title  = "Orthogonalization stability vs condition number  (n=$N_COND, k=$K_COND)",
    xscale = log10,
    yscale = log10,
)

for method in METHODS
    errs = cond_results[method]
    lines!(ax_cond, KAPPA_VALUES, errs;
        label     = string(method),
        color     = method_color[method],
        linewidth = 2,
    )
    scatter!(ax_cond, KAPPA_VALUES, errs;
        color      = method_color[method],
        markersize = 8,
    )
end

Legend(fig_cond[1, 2], ax_cond, "Method")
save("stability_vs_cond$(SUFFIX).benchmark.pdf", fig_cond)
println("Saved stability_vs_cond$(SUFFIX).benchmark.pdf")
