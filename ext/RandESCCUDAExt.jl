### Code only loaded when CUDA is as well
module RandESCCUDAExt
using LinearAlgebra
using CUDA
using RandESC
using Random
using CUDA.CUSPARSE

function RandESC.random_matrix(T::Type, m::Int, n::Int, template::CuArray; seed=Random.default_rng())
    return CUDA.randn(seed, T, m, n)
end

function RandESC.sparse_matrix(m::Int, n::Int, colptr::CuVector{<:Integer},
                               rowval::CuVector{<:Integer}, nzval::CuVector)
    return CuSparseMatrixCSC(colptr, rowval, nzval, m, n)
end

end # module