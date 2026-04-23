# sketch.jl
using Random, LinearAlgebra, SparseArrays, FFTW

# ── Sketch operator types ──────────────────────────────────────────────────
# Wrapping the sketch matrix in a struct lets us define mul!(Y, Theta, X),
# which writes the sketch in-place without allocating a temporary.

struct MatrixSketchOp{M<:AbstractMatrix}
    S::M
end
(op::MatrixSketchOp)(X) = op.S * X
LinearAlgebra.mul!(Y::AbstractVecOrMat, op::MatrixSketchOp, X::AbstractVecOrMat) = mul!(Y, op.S, X)

struct SRTTSketchOp
    diag_sign::Vector
    IX::Vector{Int}
    field::Symbol
    scale::Float64
end
(op::SRTTSketchOp)(X) = op.scale * SRTT(op.diag_sign, op.IX, X, op.field)
# FFT inside SRTT always allocates; mul! avoids the outer alloc but not the FFT temp
function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::SRTTSketchOp, X::AbstractVecOrMat)
    Y .= op(X)
    return Y
end

"""
    sketch(n::Integer, s::Integer, type::AbstractString; seed::Union{Nothing,Integer}=nothing)

Return a callable `SS` that maps `X` (size n×p) to an s×p sketch, following MATLAB's `all_sketch`.

`type` ∈ {"real_gaussian","complex_gaussian","complex_srtt","real_srtt","sparse_sign","sparse_stack"}.

- "sparse" uses the original `sparsesign` (ζ distinct rows per column, iid ±1/√ζ).
- "sparse_stack" uses `sparsestack` (blocked one-per-block construction matching the MEX).

If `seed` is provided, a fresh RNG is created with that seed (like MATLAB's `rng(seed)`).
"""
function sketch(n::Integer, s::Integer, type::AbstractString; seed::Union{Nothing,Integer}=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    n = Int(n); s = Int(s)

    t = lowercase(type)
    if t == "real_gaussian"
        S = randn(rng, s, n) / sqrt(s)
        return MatrixSketchOp(S)

    elseif t == "complex_gaussian"
        S = randn(rng, s, n) / sqrt(2s) .+ im * randn(rng, s, n) / sqrt(2s)
        return MatrixSketchOp(S)

    elseif t == "complex_srtt"
        IX = randperm(rng, n)[1:s]
        diag_sign = exp.(im .* (2π) .* rand(rng, n))
        return SRTTSketchOp(diag_sign, IX, :complex, sqrt(n/s))

    elseif t == "real_srtt"
        IX = randperm(rng, n)[1:s]
        diag_sign = rand(rng, (-1.0, 1.0), n)
        return SRTTSketchOp(diag_sign, IX, :real, sqrt(n/s))

    elseif t == "sparsesign"
        ζ = 8
        return MatrixSketchOp(sparsesign(s, n, ζ; rng))

    elseif t == "sparsestack"
        ζ = 4
        return MatrixSketchOp(sparsestack(s, n, ζ; rng))

    else
        throw(ArgumentError("Unknown sketch.type \"$type\""))
    end
end

# Subsampled Random Trig Transform (SRFT/SRDT variants)
#   complex: Y = fft(diag_sign .* X, dims=1);
#   real   : Y = dct(diag_sign .* X, 2; dims=1);
# and then subsample rows: Y = Y[IX, :].
function SRTT(diag_sign::AbstractVector, IX::AbstractVector{<:Integer},
              X::AbstractMatrix, field::Symbol)
    @assert length(diag_sign) == size(X,1) "diag_sign length must match size(X,1)"
    Xscaled = diag_sign .* X

    if field === :complex
        Y = fft(Xscaled, 1)
    elseif field === :real
        Y = FFTW.dct(Xscaled, 1)  # DCT-II along rows
    else
        throw(ArgumentError("field must be :complex or :real"))
    end
    return @view Y[IX, :]
end

# -------------------------------------------------------------------------
# Original "sparsesign" (distinct rows per column, iid ±1/√ζ)
# -------------------------------------------------------------------------
"""
    sparsesign(d::Integer, m::Integer, ζ::Integer; rng=Random.default_rng())

Build a `d × m` sparse matrix with exactly `ζ` nonzeros per column:
- For each column, choose `ζ` distinct row indices uniformly at random from `1:d`.
- Nonzeros are iid ±1/√ζ (Rademacher).

"""
function sparsesign(d::Integer, m::Integer, ζ::Integer; rng=Random.default_rng())
    d = Int(d); m = Int(m); ζ = Int(min(ζ, d))
    @assert d ≥ 1 && m ≥ 1 && ζ ≥ 1 "d, m, ζ must be positive and ζ ≤ d"

    nnz = m * ζ
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)
    v = 1.0 / sqrt(ζ)

    idx = 1
    @inbounds for j in 1:m
        rows = randperm(rng, d)[1:ζ]                 # distinct rows
        signs = rand(rng, (-1.0, 1.0), ζ)            # ±1
        for t in 1:ζ
            I[idx] = rows[t]
            J[idx] = j
            V[idx] = signs[t] * v
            idx += 1
        end
    end
    return sparse(I, J, V, d, m)
end

"""
    sparsestack(d::Integer, m::Integer, ζ::Integer; rng=Random.default_rng())

Build a `d × m` sparse matrix `S` with exactly `ζ` nonzeros **per column**
using the blocked one-per-block scheme (MEX `sparseStackl.c`):

- Partition rows `1:d` into `ζ` contiguous blocks as evenly as possible:
    `d = q*ζ + r`. First `r` blocks have size `q+1`, remaining have size `q`.
- In each column, pick ONE row uniformly from EACH block (independent across blocks/columns).
- Nonzeros are ±1/√ζ. Signs are drawn by reusing bits from random words for speed.

This mirrors the MEX behavior closely and constructs CSC arrays directly.
"""
function sparsestack(d::Integer, m::Integer, ζ::Integer; rng=Random.default_rng())
    dI = Int(d); mI = Int(m); ζI = Int(ζ)
    if dI == 0 || mI == 0 || ζI == 0
        return spzeros(dI, mI)
    end
    ζI = min(ζI, dI)

    # guard overflow in nnz = m*ζ
    if mI > typemax(Int) ÷ ζI
        throw(OverflowError("m*ζ overflows Int"))
    end
    nnz = mI * ζI

    # Partition rows: d = q*ζ + r, first r blocks of size q+1, rest q
    q, r = divrem(dI, ζI)
    sizes  = Vector{Int}(undef, ζI)
    starts = Vector{Int}(undef, ζI)
    @inbounds for j in 0:ζI-1
        if j < r
            sizes[j+1]  = q + 1
            starts[j+1] = j * (q + 1)
        else
            sizes[j+1]  = q
            starts[j+1] = r * (q + 1) + (j - r) * q
        end
    end

    # Preallocate CSC arrays
    colptr = Vector{Int}(undef, mI + 1)
    rowval = Vector{Int}(undef, nnz)
    nzval  = Vector{Float64}(undef, nnz)

    # Column pointers (1-based)
    colptr[1] = 1
    @inbounds for col in 1:mI
        colptr[col + 1] = col * ζI + 1
    end

    a = 1 / sqrt(Float64(ζI))     # magnitude
    sign_buf::UInt64 = 0x00
    bits_left::Int   = 0

    # Fill columns: each column occupies rowval/nzval indices p0+1 : p0+ζI
    @inbounds for col in 1:mI
        p0 = (col - 1) * ζI
        for j in 1:ζI
            # unbiased offset in [0, sizes[j)-1]
            off = rand(rng, 0:(sizes[j] - 1))
            rowval[p0 + j] = starts[j] + off + 1   # to 1-based

            # reuse bits from a 64-bit random word for ± sign
            if bits_left == 0
                sign_buf = rand(rng, UInt64)
                bits_left = 64
            end
            s = if (sign_buf & 0x1) == 0x1 1.0 else -1.0 end
            sign_buf >>= 1
            bits_left -= 1

            nzval[p0 + j] = s * a
        end
    end

    return SparseMatrixCSC(dI, mI, colptr, rowval, nzval)
end
