using FFTW
using GPUArraysCore

"""Enable NVTX profiling"""
function activate_nvtx_profiling()
    if isnothing(Base.get_extension(RandESC, :RandESCNVTXExt))
        @warn("NVTX and CUDA modules not loaded. NVTX profiling is not available.")
    else
        sync_nvtx_profiling(true)
    end
end
sync_nvtx_profiling() = nothing  # default no-op implementation; overridden by RandESCNVTXExt

"""Return the 2-norm of each column of `X` as a vector."""
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

"""Return the dot product of matching columns of `A` and `B` as a vector."""
function columnwise_dots(A::AbstractArray, B::AbstractArray)
    vec(sum(conj(A) .* B; dims=1))
end

"""Moves a CPU array to the same device as a template GPU array"""
function to_device(array::Array, template::AbstractGPUArray)
    dest = similar(template, eltype(array), size(array))
    copyto!(dest, array)
    return dest
end

function to_device(array::Array, template::Union{Nothing,Array})
    return array
end

"""Moves an array from the GPU to the CPU"""
function to_cpu(array::AbstractGPUArray)
    return Array(array)
end

function to_cpu(array::Array)
    return array
end

"""Generates a random matrix of a given type and size."""
function random_matrix(T::Type, m::Int, n::Int, template::Union{Array,Nothing})
    return randn(T, m, n)
end

"""Generates a random vector of a given type and size. An RNG can be optionally provided."""
function random_vector(T::Type, n::Int, template::Union{Array,Nothing})
    return randn(T, n)
end

"""Returns a CSC sparse matrix given its dimensions and the standard CSC format vectors. The sparse matrix
   lives on the same device as the template array."""
function sparse_matrix_csc(m::Int, n::Int, colptr::Vector{<:Integer},
                           rowval::Vector{<:Integer}, nzval::Vector,
                           template::Union{Array,Nothing})
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

"""Returns a CSC sparse matrix given its dimensions and the standard COO format vectors. The sparse matrix
   lives on the same device as the template array."""
function sparse_matrix_csc(I::Vector{<:Integer}, J::Vector{<:Integer},
                           V::Vector, m::Int, n::Int, template::Union{Array,Nothing})
    return sparse(I, J, V, m, n) # defaults to SparseMatrixCSC on the CPU
end

"""Returns a sparse matrix of zeros on the same device as the template array."""
function sparse_zeros(m::Int, n::Int, template::Union{Array,Nothing})
    return spzeros(m, n)
end


"""Performs a 2D discrete cosine transform (DCT type II) along the specified dimensions of the input array."""
function dct2(X::AbstractArray, dims)
    FFTW.dct(X, dims)
end