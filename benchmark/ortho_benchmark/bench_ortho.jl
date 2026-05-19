using RandESC
using LinearAlgebra
using Statistics
using CairoMakie

const METHODS = [:mgs, :mgs2, :qr, :cholqr, :cholqr2]
const N_VALUES = [500, 1000, 2000, 5000, 10000, 20000, 50000]
const N_REPS   = 5   # repetitions per (method, n); take median

function bench_method(method::Symbol, n::Int, k::Int)
    times = Vector{Float64}(undef, N_REPS)
    orth_err = 0.0

    for r in 1:N_REPS
        V   = randn(n, k)
        buf = similar(V)
        t   = @elapsed RandESC._ortho!(V, k, method, buf)
        times[r] = t
        if r == N_REPS
            # orthogonality error on a fresh random matrix
            V2 = randn(n, k)
            RandESC._ortho!(V2, k, method, similar(V2))
            orth_err = norm(V2' * V2 - I(k))
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
save("time_vs_n.pdf", fig_time)
println("\nSaved time_vs_n.pdf")

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
save("stability_vs_n.pdf", fig_stab)
println("Saved stability_vs_n.pdf")
