"""Return the 2-norm of each column of `X` as a vector."""
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

"""Return the dot product of matching columns of `A` and `B` as a vector."""
function columnwise_dots(A::AbstractArray, B::AbstractArray)
    vec(sum(conj(A) .* B; dims=1))
end

#TODO: docstrings
# Calculate the norms of the columns of an array
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

# Calculate the dot poroducts of the columns of two arrays
function columnwise_dots(A::AbstractArray, B::AbstractArray)
    vec(sum(conj(A) .* B; dims=1))
end

# Moves a CPU array to the same device as a template array (e.g. GPU array)
function to_device(array::Array, template::AbstractGPUArray)
    dest = similar(template, eltype(array), size(array))
    copyto!(dest, array)
    return dest
end

function to_device(array::Array, template::Array)
    return array
end

function to_device(array::AbstractArray, template=nothing)
    return array
end
