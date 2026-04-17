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
function sketch(n::Integer, s::Integer, type::AbstractString; seed::Union{Nothing,Integer}=nothing, template=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    n = Int(n); s = Int(s)

    # TODO: GPU porting, start with only MatrixSketchOp, because it should be easy. Then, once
    #       the solver works, we can try and look into the SRTT
    #       Once everything works, make sure docstrings and comments are up to date
    # TODO: need to pass the types in a nicer way, probably parametrically?

    t = lowercase(type)
    if t == "real_gaussian"
        S = random_matrix(Float64, s, n, template) ./ sqrt(s) # TODO: type should be inferred
        return MatrixSketchOp(S)

    elseif t == "complex_gaussian"
        #TODO: type should be inferred
        S = random_matrix(ComplexF64, s, n, template) ./ sqrt(2s)
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
        return MatrixSketchOp(sparsesign(s, n, ζ; rng, template))

    elseif t == "sparsestack"
        ζ = 4
        return MatrixSketchOp(sparsestack(s, n, ζ; rng, template))

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
function sparsesign(d::Integer, m::Integer, ζ::Integer;
                    rng=Random.default_rng(), template=nothing)
    @assert d ≥ 1 && m ≥ 1 && ζ ≥ 1 "d, m, ζ must be positive and ζ ≤ d"

    # Note: this code can be slightly optimized to run on the GPU, but even then it 
    #       remains slow. The process of generating a random permutation of rows for
    #       each column is inherently super expensive. We could be a lot more efficient
    #       if we simply picked random integers in [1, d] (small risk of suplicates).
    #       Would that be legal though?
    if template isa AbstractGPUArray
        @warn("sparsesign sketching does not run optimally on the GPU, "*
              "consider switching to sparsestack for better performance.")
    end

    nnz = m * ζ
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{ComplexF64}(undef, nnz)
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

    return sparse_matrix_csc(I, J, V, d, m, template)
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
function sparsestack(d::Integer, m::Integer, ζ::Integer; 
                     rng=Random.default_rng(), template=nothing)
    if d == 0 || m == 0 || ζ == 0
        return spzeros(d, m) #TODO: return spzeors on approriate arch
    end
    ζ = min(ζ, d)

    # guard overflow in nnz = m*ζ
    if m > typemax(Int) ÷ ζ
        throw(OverflowError("m*ζ overflows Int"))
    end
    nnz = m * ζ

    # Partition rows: d = q*ζ + r, first r blocks of size q+1, rest q
    q, r = divrem(d, ζ)
    #TODO: tiny arrays, might be simpler to build on CPU and move to GPU
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
        off = rand(0:(sizes[j] - 1)) #TODO: cannot pass rng to kernel. If crucial, then needs to be pre-generated
        starts[j] + off + 1 # to 1-based
    end

    a = 1 / sqrt(ComplexF64(ζ))     # magnitude #TODO: need a complex matrix for proper CuBLAS mul!, we really need type consistency
    #TODO: are initial random properties kept (weird random word)?
    signs = map(i -> ifelse(rand(Bool), 1, -1), indices)
    nzval = signs .* a #TODO: save an allocation by merging kernels?

    return sparse_matrix_csc(d, m, colptr, rowval, nzval, template)
end
