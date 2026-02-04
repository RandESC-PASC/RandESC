using LinearAlgebra
using Printf

# Assumes sketch.jl is included for sketch() function

"""
    jdsym_sketched(A; k=5, opts...)

Jacobi-Davidson with sketched orthogonalization for symmetric/Hermitian matrices.
Based on Balabanov-Grigori RGS framework. V is kept Θ-orthonormal,
meaning SV = Θ*V has ℓ₂-orthonormal columns.

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenvalues to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance
- `maxit=200`: Maximum iterations  
- `jmin=-1`: Min subspace dimension (default: k+5)
- `jmax=-1`: Max subspace dimension (default: jmin+5)
- `v0=nothing`: Initial vector or matrix of vectors (columns are initial vectors)
- `sigma=:SR`: Target: `:SR` (smallest real), `:LR` (largest real),
               `:SM` (smallest magnitude), `:LM` (largest magnitude)
- `M=nothing`: Preconditioner
- `disp=false`: Display progress
- `sketch_type="sparsestack"`: Sketch type (see sketch.jl)
- `sketch_size=-1`: Sketch dimension (default: 4*jmax)
- `orth_method=:rgs`: Orthogonalization method (:rcgs, :rcgs2, :rgs)

# Returns
- `(X_converged, Lambda_converged, history)`: Eigenvectors, eigenvalues, convergence history
"""
function jdsym_rand(A; k::Int=5,
                    tol::Float64=1e-8,
                    maxit::Int=200,
                    jmin::Int=-1,
                    jmax::Int=-1,
                    v0=nothing,
                    sigma::Symbol=:SR,
                    M=nothing,
                    precond_preparator=nothing,
                    disp::Bool=false,
                    sketch_type::String="sparsestack",
                    sketch_size::Int=-1,
                    orth_method::Symbol=:rgs)

    n = size(A, 1)
    k = min(k, n)
    
    if n < 1
        return zeros(ComplexF64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    # Set defaults
    jmin = jmin < 0 ? min(n, k + 5) : jmin
    jmax = jmax < 0 ? min(n, jmin + 5) : jmax
    s = sketch_size < 0 ? max(4 * jmax, 4 * k) : sketch_size

    # Normalize tolerance
    tol = tol / sqrt(k)

    # Infer element type
    if !isnothing(v0)
        T = eltype(v0)
    else
        T = ComplexF64
    end

    # Create sketch operator
    Theta = sketch(n, s, sketch_type)

    # Initialize storage for converged eigenpairs (Θ-orthonormal)
    X_converged = zeros(T, n, k)
    SX_converged = zeros(T, s, k)  # Sketched converged vectors
    AX_converged = zeros(T, n, k)  # A * X_converged for refinement
    Lambda_converged = zeros(Float64, k)
    
    # Counters
    nmv = 0
    nconv = 0
    nit = 0
    history = zeros(Float64, 0, 3)

    # Initialize search space (can be multiple vectors, Θ-orthonormal)
    V, SV = init_space_sketched(n, v0, Theta, X_converged[:, 1:nconv], SX_converged[:, 1:nconv], T, orth_method, jmax)
    j = size(V, 2)
    W = A * V
    nmv += size(V, 2)
    SW = Theta(W)

    # Sketched Rayleigh-Ritz matrix: M = (ΘV)' * (ΘAV)
    M_proj = SV' * SW

    # Main JD loop
    nlit = 0
    while nconv < k && nit < maxit
        j = size(V, 2)
        
        # Compute Ritz pairs (M is NOT symmetric even for symmetric A!)
        F = eigen(M_proj)
        lambda = F.values
        U = F.vectors
        lambda_real = real.(lambda)
        
        # Sort eigenvalues
        if sigma == :LM
            I = sortperm(abs.(lambda_real), rev=true)
        elseif sigma == :SM
            I = sortperm(abs.(lambda_real))
        elseif sigma == :LR
            I = sortperm(lambda_real, rev=true)
        else  # :SR
            I = sortperm(lambda_real)
        end
        U = U[:, I]
        lambda = lambda[I]
        
        # Best Ritz pair
        y = U[:, 1]
        y = y / norm(y)  # ℓ₂-normalize coefficient vector
        
        u = V * y
        su = SV * y
        w = W * y
        sw = SW * y
        
        # Sketched Rayleigh quotient: θ = (Θu)'(Θw) / ||Θu||²
        theta = real(dot(su, sw) / dot(su, su))
        
        # Residual
        r = w - theta * u
        sr = Theta(r)
        
        # Θ-orthogonalize residual against converged space (RGS)
        if nconv > 0
            coeffs = SX_converged[:, 1:nconv] \ sr
            r = r - X_converged[:, 1:nconv] * coeffs
            sr = sr - SX_converged[:, 1:nconv] * coeffs
        end
        
        # Convergence uses Θ-norm
        nr = norm(sr)
        
        # Record history
        history = vcat(history, [nr nit nmv])
        if disp
            @printf("it=%d, nmv=%d, dim=%d, |r|=%.2e, theta=%.6e\n", nit, nmv, j, nr, theta)
        end
        
        # Check convergence
        if nr < tol
            nconv += 1
            X_converged[:, nconv] = u
            SX_converged[:, nconv] = su
            AX_converged[:, nconv] = w
            Lambda_converged[nconv] = theta
            
            if disp
                @printf("  >>> Eigenvalue %d converged: %.10e\n", nconv, theta)
            end
            
            if nconv >= k
                break
            end
            
            # Deflate
            if j > 1
                # Project y out of remaining eigenvectors (CGS2)
                U_rest = U[:, 2:j]
                for _ in 1:2
                    U_rest = U_rest - y * (y' * U_rest)
                end
                
                # ℓ₂-orthonormalize
                Q, _ = qr(U_rest)
                Q = Matrix(Q)
                
                # Transform bases
                V = V * Q
                W = W * Q
                SV = SV * Q
                SW = SW * Q
                j = j - 1
                M_proj = Q' * M_proj * Q
            else
                # Cold restart
                V, SV = init_space_sketched(n, nothing, Theta, X_converged[:, 1:nconv], SX_converged[:, 1:nconv], T, orth_method, jmax)
                j = size(V, 2)
                W = A * V
                nmv += size(V, 2)
                SW = Theta(W)
                M_proj = SV' * SW
            end
            nlit = 0
            continue
        end
        
        # Restart if needed
        if j >= jmax
            U_keep = U[:, 1:jmin]
            Q, _ = qr(U_keep)
            Q = Matrix(Q)
            
            V = V * Q
            W = W * Q
            SV = SV * Q
            SW = SW * Q
            j = jmin
            M_proj = Q' * M_proj * Q
        end
        
        # Solve correction equation
        # Note: u needs to be Θ-normalized for projection
        nu = norm(su)
        u_proj = u / nu
        su_proj = su / nu

        Q_proj = hcat(X_converged[:, 1:nconv], u_proj)
        SQ_proj = hcat(SX_converged[:, 1:nconv], su_proj)

        t = solve_correction_sketched(Q_proj, SQ_proj, Theta, r, M, orth_method, precond_preparator, u)
        nlit += 1
        nit += 1
        
        # Expand subspace (Θ-orthogonalize)
        if nconv > 0
            t = theta_orth_against(t, X_converged[:, 1:nconv], SX_converged[:, 1:nconv], Theta, orth_method)
        end
        t = theta_orth_against(t, V, SV, Theta, orth_method)
        
        st = Theta(t)
        nt = norm(st)
        
        if nt > 1e-14
            t = t / nt
            st = st / nt
            
            wt = A * t
            nmv += 1
            swt = Theta(wt)
            
            # Update M_proj incrementally
            M_proj = [M_proj SV'*swt; st'*SW st'*swt]
            
            V = hcat(V, t)
            SV = hcat(SV, st)
            W = hcat(W, wt)
            SW = hcat(SW, swt)
        end
    end

    # Refine eigenpairs using sketched_to_fully
    X_refined, Lambda_refined = sketched_to_fully(A, X_converged[:, 1:nconv], AV=AX_converged[:, 1:nconv])

    return X_refined, Lambda_refined, history
end