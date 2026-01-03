using LinearAlgebra
using Printf

"""
    jdqr_sym(A; k=5, opts...)

Jacobi-Davidson for symmetric/Hermitian matrices (smallest/largest eigenvalues).
Direct port from MATLAB jdqr_sym_mod.m

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenvalues to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance
- `maxit=200`: Maximum iterations  
- `jmin=-1`: Min subspace dimension (default: k+5)
- `jmax=-1`: Max subspace dimension (default: jmin+5)
- `v0=nothing`: Initial vector
- `sigma=:SR`: Target: `:SR` (smallest real), `:LR` (largest real),
               `:SM` (smallest magnitude), `:LM` (largest magnitude)
- `M=nothing`: Preconditioner
- `disp=false`: Display progress

# Returns
- `(X_converged, Lambda_converged, history)`: Eigenvectors, eigenvalues, convergence history
"""
function jdsym(A; k::Int=5,
                    tol::Float64=1e-8,
                    maxit::Int=200,
                    jmin::Int=-1,
                    jmax::Int=-1,
                    v0=nothing,
                    sigma::Symbol=:SR,
                    M=nothing,
                    disp::Bool=false)

    n = size(A, 1)
    k = min(k, n)
    
    if n < 1
        return zeros(ComplexF64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    # Set defaults
    jmin = jmin < 0 ? min(n, k + 5) : jmin
    jmax = jmax < 0 ? min(n, jmin + 5) : jmax

    # Normalize tolerance
    tol = tol / sqrt(k)

    # Infer element type from initial guess or default to ComplexF64
    if !isnothing(v0)
        T = eltype(v0)
    else
        # For Hermitian matrices, we work in complex space but eigenvalues are real
        T = ComplexF64
    end

    # Initialize storage for converged eigenpairs
    X_converged = zeros(T, n, 0)
    Lambda_converged = Float64[]
    
    # Counters
    nmv = 0
    nconv = 0
    nit = 0
    history = zeros(Float64, 0, 3)

    # Initialize search space
    V = init_space(n, v0, X_converged, T)
    j = size(V, 2)
    W = A * V
    nmv += size(V, 2)
    M_proj = V' * W

    # Main JD loop
    nlit = 0
    while nconv < k && nit < maxit
        
        # Compute Ritz pairs (symmetric: M = U * D * U')
        F = eigen(Hermitian((M_proj + M_proj') / 2))
        lambda = real.(F.values)
        U = F.vectors
        
        # Sort eigenvalues
        if sigma == :LM
            I = sortperm(abs.(lambda), rev=true)
        elseif sigma == :SM
            I = sortperm(abs.(lambda))
        elseif sigma == :LR
            I = sortperm(lambda, rev=true)
        else  # :SR
            I = sortperm(lambda)
        end
        
        # Permute
        U = U[:, I]
        D = Diagonal(lambda[I])
        
        theta = D[1, 1]
        u = V * U[:, 1]
        w = W * U[:, 1]
        r = w - theta * u
        r = orth_against(r, X_converged)
        nr = norm(r)
        
        # Record history
        history = vcat(history, [nr nit nmv])
        if disp
            @printf("it=%d, nmv=%d, dim=%d, |r|=%.2e, theta=%.6e\n", nit, nmv, j, nr, real(theta))
        end
        
        # Check convergence
        if nr < tol
            X_converged = hcat(X_converged, u)
            push!(Lambda_converged, theta)
            nconv += 1
            
            if disp
                @printf("  >>> Eigenvalue %d converged: %.10e\n", nconv, real(theta))
            end
            
            if nconv >= k
                break
            end
            
            # Deflate
            if j == 1
                V = init_space(n, nothing, X_converged, T)
                j = size(V, 2)
                W = A * V
                nmv += size(V, 2)
                M_proj = V' * W
            else
                J = 2:j
                j = j - 1
                U = U[:, J]
                D = D[J, J]
                V = V * U
                W = W * U
                M_proj = D
            end
            nlit = 0
            continue
        end
        
        # Restart if needed
        if j >= jmax
            J = 1:jmin
            U = U[:, J]
            D = D[J, J]
            V = V * U
            W = W * U
            M_proj = D
            j = jmin
        end
        
        # Solve correction equation
        Q_proj = hcat(X_converged, u)
        t = solve_correction(Q_proj, r, M)
        nlit += 1
        nit += 1
        
        # Expand subspace
        t = orth_against(t, X_converged)
        t = orth_against(t, V)
        nt = norm(t)
        if nt > 1e-14
            t = t / nt
            w_new = A * t
            nmv += 1
            M_proj = [M_proj V'*w_new; t'*W t'*w_new]
            V = hcat(V, t)
            W = hcat(W, w_new)
            j += 1
        end
    end

    return X_converged, Lambda_converged, history
end


# === Helper Functions ===

function init_space(n::Int, v0, X_converged, T::Type)
    if isnothing(v0)
        V = ones(T, n, 1) .+ T(0.1) .* rand(T, n, 1)
    else
        V = T.(reshape(v0[:, 1], n, 1))
    end
    
    V = orth_against(V, X_converged)
    V, _ = qr(V)
    V = Matrix(V)
    
    return V
end


function orth_against(v, Q)
    if isempty(Q) || size(v, 1) == 0
        return v
    end
    v = copy(v)
    for _ in 1:2
        v = v - Q * (Q' * v)
    end
    return v
end


function solve_correction(Q, r, M)
    t = copy(r)
    if !isnothing(M)
        t = M \ t
    end
    t = t - Q * (Q' * t)
    return t
end