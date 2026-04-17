### Code only loaded when AMDGPU is as well
module RandESCAMDGPUExt
using LinearAlgebra
using AMDGPU
using RandESC
using Random
using AMDGPU.rocSPARSE
using SparseArrays

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
    #TODO: can we pass a RNG ?
    return AMDGPU.randn(T, m, n)
end

# AMDGPU does not support random complex matrix generation
function RandESC.random_matrix(T::Type{<:Complex}, m::Int, n::Int, template::ROCArray)
    return AMDGPU.randn(real(T), m, n) .+ im .* AMDGPU.randn(real(T), m, n)
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

end # module