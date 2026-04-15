### Code only loaded when AMDGPU is as well
module RandESCAMDGPUExt
using LinearAlgebra
using AMDGPU
using RandESC
using Random
using AMDGPU.rocSPARSE

# AMDGPU does not support non-Hermitian matrix diagonalization. Need to move to
# the CPU and back
function LinearAlgebra.eigen(A::ROCArray)
    F = eigen(Array(A))
    return ROCArray(F.values), ROCArray(F.vectors)
end

function RandESC.random_matrix(T::Type, m::Int, n::Int, template::ROCArray; seed=Random.default_rng())
    return AMDGPU.randn(seed, T, m, n)
end

function RandESC.sparse_matrix(m::Int, n::Int, colptr::ROCVector{<:Integer},
                               rowval::ROCVector{<:Integer}, nzval::ROCVector)
    return ROCSparseMatrixCSC(colptr, rowval, nzval, m, n)
end

end # module