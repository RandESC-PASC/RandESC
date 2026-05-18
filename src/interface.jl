function _default_orth_method(X0, use_randomization)
    if use_randomization
        return X0 isa AbstractGPUArray ? :rqr : :rcgs
    else
        return X0 isa AbstractGPUArray ? :qr : :mgs
    end
end

"""
    randESCSolver(A, X0, n, k; preconditioner=nothing, maxiter=100, tol=1e-6, use_randomization=false, orth_method=_default_orth_method(X0, use_randomization), verbose=false)

Solves the eigenvalue problem for the matrix `A`, computing `k` smallest eigenvalues and corresponding eigenvectors using blocked Jacobi-Davidson with the Olsen correction equation.

# Arguments
- `A`:  The matrix or linear operator for which to solve the eigenvalue problem.
- `X0`: Initial guess for the eigenvectors.
- `n`:  The size of the matrix `A`.
- `k`:  The number of eigenvalues (and eigenvectors) to compute.

# Keyword Arguments
- `preconditioner`: Preconditioner to accelerate convergence.
- `maxiter`: Maximum number of iterations allowed.
- `tol`: Tolerance for convergence.
- `precond_preparator`: Callback `f(M, X)` to refresh the preconditioner each iteration (optional).
- `use_randomization`: Boolean flag to enable or disable randomization in the solver.
- `orth_method`: Orthogonalization method to use. Defaults to `:rqr`/`:rcgs` when `use_randomization=true` and `:qr`/`:mgs` otherwise, depending on whether `X0` is a GPU array.
- `verbose`: Whether to print verbose output during the computation (default: `false`).
- `sketch_prec`: real floating-point type for sketch arrays when `use_randomization=true` (`Float32` or `Float64`; complex inputs use the corresponding complex type automatically); default: `Float32`.

# Returns
A named tuple with the following fields:
- `λ`: Computed eigenvalues.
- `X`: Computed eigenvectors.
- `residual_norms`: Norms of the residuals for each computed eigenpair.
- `n_iter`: Number of iterations performed.
- `converged`: Boolean indicating whether the solver converged.
- `n_matvec`: Number of matrix-vector products performed.
"""
function randESCSolver(A, X0::AbstractArray, n::Integer, k::Integer; preconditioner=nothing, precond_preparator=nothing, maxiter=100, tol=1e-6, use_randomization=false, orth_method=_default_orth_method(X0, use_randomization), verbose=false, sketch_prec::Type=Float32)
    if size(X0, 1) != n
        error("X0 must have $n rows (got $(size(X0, 1)))")
    end
    if size(X0, 2) != k
        error("X0 must have $k columns (got $(size(X0, 2)))")
    end

    if use_randomization
        V, lambda, history = jd_sketched(A, X0; k=k, maxit=maxiter, tol=tol, M=preconditioner, precond_preparator=precond_preparator, disp=verbose, orth_method=orth_method, sketch_prec=sketch_prec)
    else
        V, lambda, history = jd(A, X0; k=k, maxit=maxiter, tol=tol, M=preconditioner, precond_preparator=precond_preparator, disp=verbose, orth_method=orth_method)
    end
    n_iter = size(history, 1)
    converged = n_iter < maxiter
    n_matvec = isempty(history) ? 0 : Int(history[end, 3])
    if !converged
        @warn "jd did not converge within $maxiter iterations."
    end
    residual_norms = isempty(history) ? Float64[] : history[:, 1]
    return (λ=lambda, X=V, residual_norms=residual_norms, n_iter=n_iter, converged=converged, n_matvec=n_matvec)
end
