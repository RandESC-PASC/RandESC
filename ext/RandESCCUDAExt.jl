### Code only loaded when CUDA is as well
module RandESCCUDAExt
using LinearAlgebra
using CUDA
using RandESC
using Random
using CUDA.CUSPARSE
using SparseArrays
using AbstractFFTs

function RandESC.random_matrix(T::Type, m::Int, n::Int, template::CuArray)
    #TODO: can we pass a RNG ?
    return CUDA.randn(T, m, n)
end

function RandESC.random_vector(T::Type, n::Int, template::CuArray)
    #TODO: can we pass a RNG ?
    return CUDA.randn(T, n)
end

function RandESC.sparse_matrix_csc(m::Int, n::Int, colptr::CuVector{<:Integer},
                                   rowval::CuVector{<:Integer}, nzval::CuVector,
                                   template::CuArray)
    return CuSparseMatrixCSC(colptr, rowval, nzval, (m, n))
end


function RandESC.sparse_matrix_csc(I::Vector{<:Integer}, J::Vector{<:Integer},
                                   V::Vector, m::Int, n::Int, template::CuArray)
    CuSparseMatrixCSC(sparse(I, J, V, m, n)) # construct on CPU and move to GPU
end

# Emulate a descrete cosine transform using a FFT on [X; reverse(X)] whith approriate scaling
function RandESC.dct2(X::CuArray{T}, dims) where {T}
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

    indices = CuArray(collect(1:N))
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