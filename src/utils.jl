using FFTW
using GPUArraysCore

"""Return the 2-norm of each column of `X` as a vector."""
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

"""Return the dot product of matching columns of `A` and `B` as a vector."""
function columnwise_dots(A::AbstractArray, B::AbstractArray)
    vec(sum(conj(A) .* B; dims=1))
end

#TODO: docstrings
# Moves a CPU array to the same device as a template array (e.g. GPU array)
function to_device(array::Array, template::AbstractGPUArray)
    dest = similar(template, eltype(array), size(array))
    copyto!(dest, array)
    return dest
end

function to_device(array::Array, template::Union{Nothing,Array})
    return array
end

function random_matrix(T::Type, m::Int, n::Int, template::Union{Array,Nothing}; seed=Random.default_rng())
    return randn(seed, T, m, n)
end

function random_vector(T::Type, n::Int, template::Union{Array,Nothing}; seed=Random.default_rng())
    return randn(seed, T, n)
end

# Moves an array from the GPU to the CPU
function to_cpu(array::AbstractGPUArray)
    return Array(array)
end

function to_cpu(array::Array)
    return array
end

function sparse_matrix_csc(m::Int, n::Int, colptr::Vector{<:Integer},
                           rowval::Vector{<:Integer}, nzval::Vector,
                           template::Union{Array,Nothing})
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function sparse_matrix_csc(I::Vector{<:Integer}, J::Vector{<:Integer},
                           V::Vector, m::Int, n::Int, template::Union{Array,Nothing})
    return sparse(I, J, V, m, n) # defaults to SparseMatrixCSC on the CPU
end

function dct2(X::AbstractArray, dims)
    FFTW.dct(X, dims)  # DCT-II along specified dimensions
end