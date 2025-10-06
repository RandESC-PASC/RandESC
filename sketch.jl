# sketch.jl
using Random, LinearAlgebra, SparseArrays, FFTW

"""
    sketch(n::Integer, s::Integer, type::AbstractString; seed::Union{Nothing,Integer}=nothing)

Return a callable `SS` that maps `X` (size n×p) to an s×p sketch, following MATLAB's `all_sketch`.

`type` ∈ {"real_gaussian","complex_gaussian","complex_srtt","real_srtt","sparse"}.

If `seed` is provided, a fresh RNG is created with that seed (like MATLAB's `rng(seed)`).
"""
function sketch(n::Integer, s::Integer, type::AbstractString; seed::Union{Nothing,Integer}=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    n = Int(n); s = Int(s)

    t = lowercase(type)
    if t == "real_gaussian"
        # S ~ N(0, 1/s), real
        S = randn(rng, s, n) / sqrt(s)
        SS = (X -> S * X)

    elseif t == "complex_gaussian"
        # S entries have E|S_ij|^2 = 1/s (split real/imag equally)
        S = randn(rng, s, n) / sqrt(2s) .+ im * randn(rng, s, n) / sqrt(2s)
        SS = (X -> S * X)

    elseif t == "complex_srtt"
        IX = randperm(rng, n)[1:s]               # randsample(n,s)
        diag_sign = exp.(im .* (2π) .* rand(rng, n))          # length-n complex phases
        SS = (X -> sqrt(1/s) * SRTT(diag_sign, IX, X, :complex)) # should be sqrt(n/s)?

    elseif t == "real_srtt"
        IX = randperm(rng, n)[1:s]
        # simpler equivalent (vectorized):
        diag_sign = rand(rng, (-1.0, 1.0), n)
        SS = (X -> sqrt(n/s) * SRTT(diag_sign, IX, X, :real))

    elseif t == "sparse"
        # Your earlier sparsesign: s×n with zeta=8 (as in MATLAB)
        S = sparsesign(s, n, 8; rng)
        SS = (X -> S * X)

    else
        throw(ArgumentError("Unknown sketch.type \"$type\""))
    end

    return SS
end

# Subsampled Random Trig Transform (SRFT/SRDT variants)
# Matches MATLAB:
#   if complex: Y = fft(diag_sign .* X);
#   else      : Y = dct(diag_sign .* X);
#   Y = Y(IX, :);
function SRTT(diag_sign::AbstractVector, IX::AbstractVector{<:Integer},
              X::AbstractMatrix, field::Symbol)
    @assert length(diag_sign) == size(X,1) "diag_sign length must match size(X,1)"
    # Multiply by diagonal: row-wise scaling
    Xscaled = diag_sign .* X

    if field === :complex
        # FFT
        Y = fft(Xscaled,1)
    elseif field === :real
        # MATLAB dct is DCT-II by default
        Y = FFTW.dct(Xscaled, 2)
    else
        throw(ArgumentError("field must be :complex or :real"))
    end
    return @view Y[IX, :]
end

# Builds a d×m sparse ±1/√zeta matrix with zeta distinct nonzeros per column.
# adapted from MEX code by Ethan Epperly
function sparsesign(d::Integer, m::Integer, zeta::Integer; rng=Random.default_rng())
    d = Int(d); m = Int(m); zeta = Int(min(zeta, d))
    @assert d ≥ 1 && m ≥ 1 && zeta ≥ 1 "d, m, zeta must be positive; zeta ≤ d"

    nnz = m * zeta
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)
    v = 1.0 / sqrt(zeta)

    idx = 1
    for j in 1:m
        rows = randperm(rng, d)[1:zeta]   # distinct rows per column
        signs = rand(rng, (-1.0, 1.0), zeta)          # ±1
        @inbounds for t in 1:zeta
            I[idx] = rows[t]
            J[idx] = j
            V[idx] = signs[t] * v
            idx += 1
        end
    end
    return sparse(I, J, V, d, m)
end