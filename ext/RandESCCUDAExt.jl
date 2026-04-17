### Code only loaded when CUDA is as well
module RandESCCUDAExt
using LinearAlgebra
using CUDA
using RandESC
using Random
using CUDA.CUSPARSE
using SparseArrays

function RandESC.random_matrix(T::Type, m::Int, n::Int, template::CuArray)
    #TODO: can we pass a RNG ?
    return CUDA.randn(T, m, n)
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

end # module