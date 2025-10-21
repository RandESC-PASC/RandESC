# using Classes
using LinearMaps

# @class AlgorithmVariant <: Class begin
#     name:: String
#     description:: String
#     preconditioonPossible:: Bool
#     randomizedVersionImplemented:: Bool
#     blockedInitialGuessPossible:: Bool
#     function AlgorithmVariant(name:: String, description:: String, preconditioonPossible:: Bool, randomizedVersionImplemented:: Bool, blockedInitialGuessPossible:: Bool)
#         new(name, description, preconditioonPossible, randomizedVersionImplemented, blockedInitialGuessPossible)
#     end
# end

# const IRA = AlgorithmVariant("IRA", "Implicitly Restarted Arnoldi", false, true, false)
# const LOBPCG = AlgorithmVariant("LOBPCG", "Locally Optimal Block Preconditioned Conjugate Gradient", true, false, true)

# algorithms = [IRA, LOBPCG]

# @class EigenSolver <: Class begin

#     name:: String
#     usePreconditioner:: Bool
#     useRandomizedVersion:: Bool
#     useBlockedInitialGuess:: Bool

#     function EigenSolver(name:: String, usePreconditioner:: Bool, useRandomizedVersion:: Bool, useBlockedInitialGuess:: Bool)
#         # check if the combination is valid
#         for alg in algorithms
#             if alg.name == name
#                 if usePreconditioner && !alg.preconditioonPossible
#                     error("Algorithm $name does not support preconditioning.")
#                 elseif useRandomizedVersion && !alg.randomizedVersionImplemented
#                     error("Algorithm $name does not have a randomized version implemented.")
#                 elseif useBlockedInitialGuess && !alg.blockedInitialGuessPossible
#                     error("Algorithm $name does not support blocked initial guesses.")
#                 else
#                     return new(name, usePreconditioner, useRandomizedVersion, useBlockedInitialGuess)
#                 end
#             end
#         end
#         error("Unknown algorithm name: $name")
#     end
# end


"""
    randESCSolver(A, n, k; X0, preconditioner, maxiter, tol, useRandomization, method, orth_method="cgs2", cleanEigenvectors=false, normA::float, verbose=false)

Solves the eigenvalue problem for the matrix `A`, computing `k` eigenvalues and corresponding eigenvectors.

# Arguments
- `A`: The matrix or linear operator for which to solve the eigenvalue problem.
- `n`: The size of the matrix `A`.
- `k`: The number of eigenvalues (and eigenvectors) to compute.

# Keyword Arguments
- `X0`: Initial guess for the eigenvectors.
- `preconditioner`: Preconditioner to accelerate convergence.
- `maxiter`: Maximum number of iterations allowed.
- `tol`: Tolerance for convergence.
- `useRandomization`: Boolean flag to enable or disable randomization in the solver.
- `method`: The eigensolver method to use (e.g., "arnoldi", "lanczos").
- `orth_method`: Orthogonalization method to use (default: `"depends on solver"`).
- `cleanEigenvectors`: Whether to clean the eigenvectors (make them more orthogonal) after computation (default: `false`). Useful for randomized methods.
- `normA`: An estimate of the norm of `A`, used for scaling residuals (optional).
- `verbose`: Whether to print verbose output during the computation (default: `false`).

# Returns
A named tuple with the following fields:
- `λ`: Computed eigenvalues.
- `X`: Computed eigenvectors.
- `residual_norms`: Norms of the residuals for each computed eigenpair.
- `n_iter`: Number of iterations performed.
- `converged`: Boolean indicating whether the solver converged.
- `n_matvec`: Number of matrix-vector products performed.
"""
function randESCSolver(A, n::Integer, k::Integer; X0=nothing, preconditioner=nothing, precond_preparator=nothing, maxiter=100, tol=1e-6, useRandomization=false, method="", orth_method=nothing, cleanEigenvectors=false, normA=nothing, verbose=false)
    # if !(A isa AbstractMatrix) && !(A isa LinearMap)
    #     error("A must be an AbstractMatrix or LinearMap")
    # end
    if isnothing(X0)
        X0 = randn(n, k)
    end
    if !(X0 isa AbstractMatrix)
        error("X0 must be an AbstractMatrix")
    end
    if size(X0, 1) != n
        error("X0 must have $n rows (got $(size(X0, 1)))")
    end
    if size(X0, 2) != k
        error("X0 must have $k columns (got $(size(X0, 2)))")
    end

    # solver = EigenSolver(method, preconditioner != nothing, useRandomization, size(X0, 2) > 1)
    if method == "" # auto-select method
        if preconditioner != nothing
            method = "LOBPCG"
        else
            method = "IRA"
        end
    end


    if method == "IRA"
        if preconditioner != nothing 
            @warn "Preconditioner is provided for IRA, but IRA does not support preconditioning."
            # println("Warning: Preconditioner is provided for IRA.")
            # println(preconditioner)
            # error("Preconditioning is not supported for IRA.")
        end
        if size(X0, 2) > 1
            @warn "Blocked initial guesses are provided for IRA, but IRA does not support blocked initial guesses. Using only the first column of X0."
            X0 = X0[:, 1]  # use only the first column
            # error("Blocked initial guesses are not supported for IRA.")
        end
        if useRandomization
            # call rand_ira
            V, D, ritz, info = rand_ira(A, n, k; v0=X0, maxit=maxiter, tol=tol, verbose=verbose)
            if cleanEigenvectors
                # call sketched_to_fully to clean the eigenvectors
                V, D = sketched_to_fully(A, V)
            end

            lamda = diag(D)
            n_iter = info.it
            converged = n_iter < maxiter
            n_matvec = info.mvps
            if !converged
                @warn "rand_ira did not converge within $maxiter iterations. Max residual norm: $(maximum(ritz.res))"
            end
            residual_norms = ritz.res


            return (λ=lamda, X=V, residual_norms=residual_norms, n_iter=n_iter, converged=converged, n_matvec=n_matvec)
        else
            V, D, ritz, info = ira(A, n, k; v0=X0, maxit=maxiter, tol=tol, verbose=verbose)
            lamda = diag(D)
            n_iter = info.it
            converged = n_iter < maxiter
            n_matvec = info.mvps
            if !converged
                @warn "ira did not converge within $maxiter iterations. Max residual norm: $(maximum(ritz.res))"
            end
            residual_norms = ritz.res
            return (λ=lamda, X=V, residual_norms=residual_norms, n_iter=n_iter, converged=converged, n_matvec=n_matvec)
        end
    elseif method == "LOBPCG"
        if useRandomization
            error("Randomized version is not implemented for LOBPCG.")
        else
            println("Using LOBPCG")
            println("n, k: ", n, ", ", k)
            V, lambda, info = lobpcg_softlock(A, n, k; X0=X0, maxit=maxiter, tol=tol, verbosity=verbose ? 1 : 0, M=preconditioner, normA=normA, precond_preparation=precond_preparator)

            n_iter = info.it
            converged = n_iter < maxiter
            n_matvec = info.mvps
            if !converged
                @warn "lobpcg did not converge within $maxiter iterations. Max residual norm: $(maximum(info.res))"
            end
            residual_norms = info.res
            return (λ=lambda, X=V, residual_norms=residual_norms, n_iter=n_iter, converged=converged, n_matvec=n_matvec)
        end
    else
        error("Unknown method: $method. Supported methods are: IRA, LOBPCG.")
    end
end