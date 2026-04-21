"""Return the 2-norm of each column of `X` as a vector."""
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

"""Return the dot product of matching columns of `A` and `B` as a vector."""
function columnwise_dots(A::AbstractArray, B::AbstractArray)
    vec(sum(conj(A) .* B; dims=1))
end
