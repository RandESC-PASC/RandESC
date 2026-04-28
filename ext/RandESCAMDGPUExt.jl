### Code only loaded when AMDGPU is as well
module RandESCAMDGPUExt
using LinearAlgebra
using AMDGPU
using RandESC
using Random
using AMDGPU.rocSPARSE
using SparseArrays
using AbstractFFTs

# AMDGPU does not support non-Hermitian matrix diagonalization. Need to move to
# the CPU and back. For Julia's multiple dispatch to work, we need to be more
# specialized than rocSOLVER, hence the explicit loop over types (rather than an
# union). Explicitly symmetric/Hermitian matrices will not be dispatched here.
for T in [Float32, Float64, ComplexF32, ComplexF64]
    @eval begin
        function LinearAlgebra.eigen(A::ROCMatrix{$T})
            F = eigen(Array(A))
            return (; values = ROCArray(F.values), vectors = ROCArray(F.vectors))
        end
    end
end

function RandESC.random_matrix(T::Type{<:Real}, m::Int, n::Int, template::ROCArray)
    return AMDGPU.randn(T, m, n)
end

# AMDGPU does not support random complex matrix generation
function RandESC.random_matrix(T::Type{<:Complex}, m::Int, n::Int, template::ROCArray)
    return AMDGPU.randn(real(T), m, n) .+ im .* AMDGPU.randn(real(T), m, n)
end

function RandESC.random_vector(T::Type, n::Int, template::ROCArray)
    return AMDGPU.randn(T, n)
end

# AMDGPU does not support random complex vector generation
function RandESC.random_vector(T::Type{<:Complex}, n::Int, template::ROCArray)
    return AMDGPU.randn(real(T), n) .+ im .* AMDGPU.randn(real(T), n)
end

function RandESC.sparse_matrix_csc(m::Int, n::Int, colptr::ROCVector{<:Integer},
                                   rowval::ROCVector{<:Integer}, nzval::ROCVector,
                                   template::ROCArray)
    return ROCSparseMatrixCSC(colptr, rowval, nzval, (m, n))
end

function RandESC.sparse_matrix_csc(I::Vector{<:Integer}, J::Vector{<:Integer},
                                   V::Vector, m::Int, n::Int, template::ROCArray)
    return ROCSparseMatrixCSC(sparse(I, J, V, m, n)) # construct on CPU and move to GPU
end

function RandESC.sparse_zeros(m::Int, n::Int, template::ROCArray)
    return ROCSparseMatrixCSC(spzeros(m, n)) # build on CPU and move to GPU
end

# Emulate a descrete cosine transform using a FFT on [X; reverse(X)] whith approriate scaling
function RandESC.dct2(X::ROCArray{T}, dims) where {T}
    dims != 1 && error("dct2 only implemented for dims=1 (DCT along rows)")
    N = size(X, 1)

    # Even extension: [x; reverse(x)]
    ext_size = (2N, size(X, 2))
    x_ext = similar(X, ext_size...)

    # first half
    inds1 = [1:N, :]
    x_ext[inds1...] .= X

    # reversed second half
    inds2 = [N + 1:2N, :]
    x_ext[inds2...] .= reverse(X; dims=1)

    Y = fft(x_ext, 1)

    indices = ROCArray(collect(1:N))
    scaling = map(indices) do idx
        s = 0.5 * exp(-im * π * (idx - 1) / (2N))
        idx == 1 ? s/sqrt(N) : s*sqrt(2/N)
    end
    if T <: Real
        return real.(Y[inds1...] .* scaling)
    else
        return Y[inds1...] .* scaling
    end
end

end # module