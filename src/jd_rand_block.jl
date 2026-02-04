using LinearAlgebra
using Printf

"""
    jdsym_rand_block(A; k=5, opts...)

Block Jacobi-Davidson with sketched orthogonalization for symmetric/Hermitian matrices.
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
- `blk=-1`: Block size (default: min(k, 4))
- `v0=nothing`: Initial vectors (n × m matrix)
- `sigma=:SR`: Target: `:SR` (smallest real), `:LR` (largest real),
               `:SM` (smallest magnitude), `:LM` (largest magnitude)
- `M=nothing`: Preconditioner
- `precond_preparator=nothing`: Function to prepare preconditioner
- `disp=false`: Display progress
- `sketch_type="sparsestack"`: Sketch type (see sketch.jl)
- `sketch_size=-1`: Sketch dimension (default: 4*jmax)
- `orth_method=:rgs`: Orthogonalization method (:rcgs, :rcgs2, :rgs)

# Returns
- `(X, Lambda, history)`: Eigenvectors, eigenvalues, convergence history
"""
function jdsym_rand_block(A; k::Int=5,
                          tol::Float64=1e-8,
                          maxit::Int=200,
                          jmin::Int=-1,
                          jmax::Int=-1,
                          blk::Int=-1,
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
        return zeros(Float64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    # Set defaults
    blk = blk < 0 ? min(k, 4) : blk
    jmin = jmin < 0 ? min(n, k + 5) : jmin
    jmax = jmax < 0 ? min(n, jmin + 5) : jmax
    s = sketch_size < 0 ? max(4 * jmax, 4 * k) : sketch_size

    # Normalize tolerance
    tol_eff = tol / sqrt(k)

    # Infer element type
    T = isnothing(v0) ? ComplexF64 : eltype(v0)

    # Create sketch operator
    Theta = sketch(n, s, sketch_type)

    # Converged eigenpairs (Θ-orthonormal)
    Qschur = zeros(T, n, 0)
    SQschur = zeros(T, s, 0)
    Rschur = Float64[]

    # Counters
    nmv = 0
    nconv = 0
    nit = 0
    history = zeros(Float64, 0, 3)

    # Initialize search space with blk vectors
    v0_init = isnothing(v0) ? randn(T, n, blk) : v0
    if size(v0_init, 2) < blk
        v0_init = hcat(v0_init, randn(T, n, blk - size(v0_init, 2)))
    end
    V, SV = init_space_sketched(n, v0_init, Theta, Qschur, SQschur, T, orth_method, jmax)
    j = size(V, 2)
    W = A * V
    nmv += j
    SW = Theta(W)

    # Sketched Rayleigh-Ritz matrix: M_proj = (ΘV)' * (ΘW)
    M_proj = SV' * SW

    # Main block JD loop
    while nconv < k && nit < maxit
        j = size(V, 2)

        # Solve projected eigenvalue problem
        F = eigen(M_proj)
        lambda = F.values
        U = F.vectors
        lambda_real = real.(lambda)

        # Sort by target
        if sigma == :LM
            perm = sortperm(abs.(lambda_real), rev=true)
        elseif sigma == :SM
            perm = sortperm(abs.(lambda_real))
        elseif sigma == :LR
            perm = sortperm(lambda_real, rev=true)
        else  # :SR
            perm = sortperm(lambda_real)
        end
        U = U[:, perm]
        lambda = lambda[perm]

        # Active block size
        p = min(blk, j, k - nconv)

        # Block Ritz vectors
        Y = U[:, 1:p]
        X_blk = V * Y
        SX_blk = SV * Y
        W_blk = W * Y
        theta_vals = real.(lambda[1:p])

        # Block residuals: R = W_blk - X_blk * Diagonal(θ)
        R_blk = W_blk - X_blk * Diagonal(theta_vals)
        SR_blk = Theta(R_blk)

        # Θ-orthogonalize residuals against converged space
        if nconv > 0
            @views coeffs = SQschur \ SR_blk
            R_blk = R_blk - Qschur * coeffs
            SR_blk = Theta(R_blk)
        end

        # Residual norms (Θ-norm per column)
        nr = [norm(view(SR_blk, :, i)) for i in 1:p]

        # Record history
        history = vcat(history, [minimum(nr) nit nmv])
        if disp
            @printf("it=%d, nmv=%d, dim=%d, blk=%d, |r|=[%.2e, %.2e]\n",
                    nit, nmv, j, p, minimum(nr), maximum(nr))
        end

        # Check convergence
        conv_idx = findall(nr .< tol_eff)
        if !isempty(conv_idx)
            # QR-orthonormalize converged coefficient vectors
            Y_conv = Y[:, conv_idx]
            F_qr = qr(Y_conv)
            nc = length(conv_idx)
            QY = Matrix(F_qr.Q)[:, 1:nc]
            VQY = V * QY

            Qschur = hcat(Qschur, VQY)
            append!(Rschur, theta_vals[conv_idx])
            SQschur = hcat(SQschur, Theta(VQY))
            nconv += nc

            if disp
                for (i, ci) in enumerate(conv_idx)
                    @printf("  >>> Eigenvalue %d converged: %.10e\n", nconv - nc + i, theta_vals[ci])
                end
            end

            if nconv >= k
                break
            end

            # Deflate: keep unconverged Ritz vectors
            keep_idx = setdiff(1:j, conv_idx)
            if isempty(keep_idx)
                # Cold restart
                v0_restart = randn(T, n, blk)
                V, SV = init_space_sketched(n, v0_restart, Theta, Qschur, SQschur, T, orth_method, jmax)
                j = size(V, 2)
                W = A * V
                nmv += j
                SW = Theta(W)
                M_proj = SV' * SW
            else
                U_keep = U[:, keep_idx]
                # Project out converged directions (CGS2)
                for _ in 1:2
                    U_keep = U_keep - QY * (QY' * U_keep)
                end
                nk = length(keep_idx)
                F_qr = qr(U_keep)
                Q_def = Matrix(F_qr.Q)[:, 1:nk]
                V = V * Q_def
                W = W * Q_def
                SV = SV * Q_def
                SW = SW * Q_def
                M_proj = Q_def' * M_proj * Q_def
            end
            continue
        end

        # Restart if subspace too large
        if j >= jmax
            nk = min(jmin, j)
            U_keep = U[:, 1:nk]
            F_qr = qr(U_keep)
            Q_rst = Matrix(F_qr.Q)[:, 1:nk]
            V = V * Q_rst
            W = W * Q_rst
            SV = SV * Q_rst
            SW = SW * Q_rst
            M_proj = Q_rst' * M_proj * Q_rst
        end

        # Block correction equation
        # Project out converged + active Ritz vectors
        F_qr = qr(SX_blk)
        SX_proj = Matrix(F_qr.Q)[:, 1:p]
        RY = F_qr.R
        X_proj = X_blk / RY

        Q_proj = hcat(Qschur, X_proj)
        SQ_proj = hcat(SQschur, SX_proj)

        # Solve: T = (I - Q Q'_Θ) M⁻¹ R
        T_corr = copy(R_blk)
        if !isnothing(M)
            if !isnothing(precond_preparator)
                precond_preparator(M, X_blk)
            end
            T_corr = M \ T_corr
        end
        T_corr = theta_orth_block_against(T_corr, Q_proj, SQ_proj, Theta, orth_method)
        nit += 1

        # Expand subspace: Θ-orthogonalize correction block
        if nconv > 0
            T_corr = theta_orth_block_against(T_corr, Qschur, SQschur, Theta, orth_method)
        end
        T_corr = theta_orth_block_against(T_corr, V, SV, Theta, orth_method)

        # Θ-orthonormalize correction vectors internally
        T_corr, ST_corr = theta_orth_block(T_corr, Theta, orth_method)

        if size(T_corr, 2) > 0
            W_new = A * T_corr
            nmv += size(T_corr, 2)
            SW_new = Theta(W_new)

            # Update M_proj incrementally
            M_proj = [M_proj SV'*SW_new; ST_corr'*SW ST_corr'*SW_new]
            V = hcat(V, T_corr)
            SV = hcat(SV, ST_corr)
            W = hcat(W, W_new)
            SW = hcat(SW, SW_new)
        end
    end

    # Refine eigenpairs using sketched_to_fully
    if nconv > 0
        X_refined, Lambda_refined = sketched_to_fully(A, Qschur)
        return X_refined, Lambda_refined, history
    else
        return zeros(T, n, 0), Float64[], history
    end
end
