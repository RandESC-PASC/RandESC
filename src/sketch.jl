using Random, LinearAlgebra, SparseArrays

# ── Sketch operator types ──────────────────────────────────────────────────
# Wrapping the sketch matrix in a struct lets us define mul!(Y, Theta, X),
# which writes the sketch in-place without allocating a temporary.

struct MatrixSketchOp{M<:AbstractMatrix}
    S::M
end
(op::MatrixSketchOp)(X) = op.S * X
function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::MatrixSketchOp, X::AbstractVecOrMat)
    TS = eltype(op.S)
    if eltype(X) == TS
        mul!(Y, op.S, X)
    else
        Xc = similar(X, TS)
        copyto!(Xc, X)
        mul!(Y, op.S, Xc)
    end
end

struct SRTTSketchOp
    diag_sign::AbstractVector
    IX::AbstractVector{Int}
    field::Symbol
    scale::Float32
end
(op::SRTTSketchOp)(X) = op.scale * SRTT(op.diag_sign, op.IX, X, op.field)
# FFT inside SRTT always allocates; mul! avoids the outer alloc but not the FFT temp
function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::SRTTSketchOp, X::AbstractVecOrMat)
    Y .= op(X)
    return Y
end

"""
    sketch(n::Integer, s::Integer, type::AbstractString, template=AbstractArray{T};
           seed::Union{Nothing,Integer}=nothing)

Return a callable `SS` that maps `X` (size n×p) to an s×p sketch, roughly following 
MATLAB's `all_sketch`. Architecture (CPU vs GPU) and element types are inferred from `template`.

`type` ∈ {"real_gaussian","complex_gaussian","complex_srtt","real_srtt","sparse_sign","sparse_stack"}.

- "sparse" uses the original `sparsesign` (ζ distinct rows per column, iid ±1/√ζ).
- "sparse_stack" uses `sparsestack` (blocked one-per-block construction matching the MEX).
- "real_gaussian" uses a randomly generated real Gaussian matrix, scaled by 1/√s.
- "complex_gaussian" uses a randomly generated complex Gaussian matrix, scaled by 1/√(2s).
- "complex_srtt" uses a ramdom complex Subsampled Random Trig Transform
- "real_srtt" uses a ramdom real Subsampled Random Trig Transform

If `seed` is provided, the Random module is seeded with Random.seed!(seed), guranteeing a reproducible
sequence of generated random numbers.
"""
function sketch(n::Integer, s::Integer, type::AbstractString, template::AbstractArray{T};
                seed::Union{Nothing,Integer}=nothing) where {T}
    n = Int(n); s = Int(s)
    isnothing(seed) || Random.seed!(seed)

    # Sketch arrays are always single precision; complex inputs get ComplexF32.
    ST = T <: Complex ? ComplexF32 : Float32
    template32 = similar(template, ST, 1)  # device proxy, Float32 element type

    st = lowercase(type)
    if st == "real_gaussian"
        S = random_matrix(Float32, s, n, template) ./ Float32(sqrt(s))
        return MatrixSketchOp(S)

    elseif st == "complex_gaussian"
        S = random_matrix(ComplexF32, s, n, template) ./ Float32(sqrt(2s))
        return MatrixSketchOp(S)

    elseif st == "complex_srtt"
        IX = to_device(randperm(n)[1:s], template)
        diag_sign = similar(template, ComplexF32, n)
        map!(_ -> ComplexF32(exp(im * 2π * rand(Float32))), diag_sign, diag_sign)
        return SRTTSketchOp(diag_sign, IX, :complex, Float32(1/sqrt(s)))

    elseif st == "real_srtt"
        IX = to_device(randperm(n)[1:s], template)
        diag_sign = similar(template, Float32, n)
        map!(_ -> ifelse(rand(Bool), 1f0, -1f0), diag_sign, diag_sign)
        return SRTTSketchOp(diag_sign, IX, :real, Float32(sqrt(n/s)))

    elseif st == "sparsesign"
        ζ = 8
        return MatrixSketchOp(sparsesign(s, n, ζ, template32))

    elseif st == "sparsestack"
        ζ = 4
        return MatrixSketchOp(sparsestack(s, n, ζ, template32))

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
    TS = eltype(diag_sign)
    Xc = eltype(X) == TS ? X : TS.(X)
    Xscaled = diag_sign .* Xc

    if field === :complex
        Y = fft(Xscaled, 1)
    elseif field === :real
        Y = dct2(Xscaled, 1) # DCT-II along rows
    else
        throw(ArgumentError("field must be :complex or :real"))
    end
    return Y[IX, :]
end

# -------------------------------------------------------------------------
# Original "sparsesign" (distinct rows per column, iid ±1/√ζ)
# -------------------------------------------------------------------------
"""
    sparsesign(d::Integer, m::Integer, ζ::Integer, teamplate=AbstractArray{T})

Build a `d × m` sparse matrix with exactly `ζ` nonzeros per column:
- For each column, choose `ζ` distinct row indices uniformly at random from `1:d`.
- Nonzeros are iid ±1/√ζ (Rademacher).

"""
function sparsesign(d::Integer, m::Integer, ζ::Integer,
                    template::AbstractArray{T}) where {T}
    @assert d ≥ 1 && m ≥ 1 && ζ ≥ 1 "d, m, ζ must be positive and ζ ≤ d"

    # Note: this code can be slightly optimized to run on the GPU, but even then it 
    #       remains slow. The process of generating a random permutation of rows for
    #       each column is inherently super expensive.
    if template isa AbstractGPUArray
        @warn("sparsesign sketching does not run optimally on the GPU, "*
              "consider switching to sparsestack for better performance.")
    end

    nnz = m * ζ
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{T}(undef, nnz)
    v = 1.0 / sqrt(ζ)

    idx = 1
    @inbounds for j in 1:m
        rows = randperm(d)[1:ζ]                 # distinct rows
        signs = rand((-1.0, 1.0), ζ)            # ±1
        for t in 1:ζ
            I[idx] = rows[t]
            J[idx] = j
            V[idx] = signs[t] * v
            idx += 1
        end
    end

    return sparse_matrix_csc(I, J, V, d, m, template)
end

"""
    sparsestack(d::Integer, m::Integer, ζ::Integer, template=AbstractArray{T})

Build a `d × m` sparse matrix `S` with exactly `ζ` nonzeros **per column**
using the blocked one-per-block scheme (MEX `sparseStackl.c`):

- Partition rows `1:d` into `ζ` contiguous blocks as evenly as possible:
    `d = q*ζ + r`. First `r` blocks have size `q+1`, remaining have size `q`.
- In each column, pick ONE row uniformly from EACH block (independent across blocks/columns).
- Nonzeros are ±1/√ζ. Signs are drawn by reusing bits from random words for speed.

This mirrors the MEX behavior closely and constructs CSC arrays directly.
"""
function sparsestack(d::Integer, m::Integer, ζ::Integer,
                     template::AbstractArray{T}) where {T}
    if d == 0 || m == 0 || ζ == 0
        return sparse_zeros(d, m, template)
    end
    ζ = min(ζ, d)

    # guard overflow in nnz = m*ζ
    if m > typemax(Int) ÷ ζ
        throw(OverflowError("m*ζ overflows Int"))
    end
    nnz = m * ζ

    # Partition rows: d = q*ζ + r, first r blocks of size q+1, rest q
    q, r = divrem(d, ζ)
    js = to_device(collect(0:ζ-1), template)
    sizes = map(j -> ifelse(j < r, q + 1, q), js)
    starts = map(j -> ifelse(j < r,
                             j * (q + 1),
                             r * (q + 1) + (j - r) * q), js)

    # Column pointers (1-based)
    indices = to_device(collect(1:m+1), template)
    colptr = map(col -> (col - 1) * ζ + 1, indices)

    indices = to_device(collect(1:nnz), template) # compound indices
    rowval = map(indices) do idx
        j = (idx - 1) % ζ + 1 # idx = (col - 1) * ζ + j
        # unbiased offset in [0, sizes[j)-1]
        off = rand(0:(sizes[j] - 1))
        starts[j] + off + 1 # to 1-based
    end

    a = 1 / sqrt(ζ)     # magnitude
    nzval = similar(template, T, nnz)
    map!(i -> ifelse(rand(Bool), a, -a), nzval, indices)

    return sparse_matrix_csc(d, m, colptr, rowval, nzval, template)
end
