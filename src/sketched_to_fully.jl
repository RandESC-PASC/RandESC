using LinearAlgebra

# Generic application of a linear operator A to a vector or a block of vectors.
applyA(A::AbstractMatrix, x::AbstractVecOrMat) = A * x          # fast path for matrices
applyA(A,                    x::AbstractVecOrMat) = A(x)         # fallback for callables (Function or functor)

"""
    sketched_to_fully(A, V) -> W, D

Given a (possibly implicit) linear operator `A` and a sketch/basis `V` (n×k),
orthonormalizes `V` via Cholesky, forms `F = V' * A * V` without assuming that
`A * V` is defined for non-matrix `A`, computes its eigendecomposition, and
returns:

- `W = V * Q`  (Ritz vectors),
- `D = Diagonal(λ)`  (Ritz values as a diagonal matrix).

`A` may be any `AbstractMatrix` (dense/sparse/wrapper) or a callable object
taking a vector/matrix and returning `A*x`.
"""
function sketched_to_fully(A, V)
    # Cholesky-based orthonormalization: V ← V / chol(V'V)
    K = V' * V
    R = cholesky(K).U
    Vn = V / R

    # Compute A * Vn:
    # - If A is a matrix, do it in one shot.
    # - Otherwise, apply columnwise via applyA without a helper function.
    AV = applyA(A,Vn)

    # Form the small projected matrix and eigendecompose
    F = Vn' * AV
    eigres = eigen(F)

    # Assemble outputs (match MATLAB-style: eigenvalues in a diagonal matrix)
    W = Vn * eigres.vectors
    D = Diagonal(eigres.values)
    return W, D
end

