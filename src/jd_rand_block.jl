using LinearAlgebra
using Printf

"""
    jdsym_rand_block(A; k=5, kwargs...)

Blocked Jacobi-Davidson with sketched (Θ-norm) orthogonalization for
symmetric/Hermitian matrices (smallest eigenvalues by default).
Based on Balabanov-Grigori RGS framework; V is kept Θ-orthonormal,
meaning SV = ΘV has ℓ₂-orthonormal columns.

Same structural tricks as `jdsym_block`:
  - Block size defaults to k
  - Consecutive convergence check from pair 1
  - Reuse unconverged residuals after deflation (no wasted iteration)
  - Soft locking: converged vectors stored in Qschur, projected out of
    residuals and correction vectors every iteration

Algorithm outline per iteration:
  1. Diagonalize sketched Rayleigh-Ritz matrix M = (ΘV)'(ΘAV)
     (general complex EVP; eigenvalues are approximately real for Hermitian A)
  2. Restart subspace if j + nb > jmax  (keep best jmin Ritz vectors, QR-orthonormalized)
  3. Compute nb Ritz vectors and residuals; project out converged (Θ-norm)
  4. Check consecutive convergence from pair 1; deflate converged pairs
  5. Reuse unconverged residuals; apply preconditioner; project corrections
     using sketched oblique projector against converged + active Ritz vecs
  6. Expand subspace: Θ-orthogonalize against V, update M_proj incrementally

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenpairs to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance on residual Θ-norms (‖Θr‖)
- `maxit=200`: Maximum outer iterations
- `blk=-1`: Block size (default: `k`)
- `jmin=-1`: Subspace size after restart (default: `k + 5`)
- `jmax=-1`: Max subspace dimension (default: `jmin + 10`)
- `v0=nothing`: Initial vectors (n × ≤k matrix)
- `sigma=:SR`: Target: `:SR` smallest real, `:LR` largest real,
               `:SM` smallest magnitude, `:LM` largest magnitude
- `M=nothing`: Preconditioner (applied as `M \\ r`)
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info
- `sketch_type="sparsestack"`: Sketch operator type (see sketch.jl)
- `sketch_size=-1`: Sketch dimension s (default: `max(4*jmax, 4*k)`)
- `orth_method=:rgs`: Θ-orthogonalization method (`:rgs`, `:rcgs`, `:rcgs2`)

# Returns
`(X, lambda, history)` where
- `X` (n × k): eigenvectors
- `lambda` (k,): eigenvalues
- `history` (nit × 3): columns are [max_rnorm, iter, nmv]
"""
@timing "jdsym_rand_block" function jdsym_rand_block(A; k::Int=5,
                          tol::Float64=1e-8,
                          maxit::Int=200,
                          blk::Int=-1,
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
        return zeros(Float64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    # Block size = k (main trick from jd_block.jl)
    blk  = blk  < 0 ? k              : blk
    jmin = jmin < 0 ? min(n, k + 5)  : jmin
    jmax = jmax < 0 ? min(n, jmin + 10) : jmax
    s    = sketch_size < 0 ? max(4jmax, 4k) : sketch_size

    T = isnothing(v0) ? ComplexF64 : eltype(v0)

    Theta = sketch(n, s, sketch_type)

    # Converged eigenpairs (kept Θ-orthonormal: SQschur'*SQschur = I)
    Qschur  = zeros(T, n, 0)
    SQschur = zeros(T, s, 0)
    lambda_conv = Float64[]

    nmv   = 0
    nconv = 0
    nit   = 0
    history = zeros(Float64, 0, 3)

    # ── Initialize subspace ───────────────────────────────────────────────
    v0_init = isnothing(v0) ? randn(T, n, blk) : v0
    if size(v0_init, 2) < blk
        v0_init = hcat(v0_init, randn(T, n, blk - size(v0_init, 2)))
    end
    V, SV = init_space_sketched(n, v0_init, Theta, Qschur, SQschur, T, orth_method, jmax)
    j = size(V, 2)

    @timing "jdsym_rand_block: matvec" begin
        W  = A * V
        nmv += j
    end
    SW = Theta(W)

    # M_proj = (ΘV)'*(ΘAV); general complex matrix (approximately Hermitian for Hermitian A)
    @timing "jdsym_rand_block: overlap" M_proj = SV' * SW

    # ── Main loop ──────────────────────────────────────────────────────────
    for iter in 1:maxit
        nit = iter
        j   = size(V, 2)

        # Diagonalize projected EVP (general eigen; M_proj only approximately Hermitian)
        @timing "jdsym_rand_block: diag" begin
            F  = eigen(M_proj)
            ew = real.(F.values)
            U  = F.vectors
            perm = _jdrb_sort_perm(ew, sigma)
            ew = ew[perm]
            U  = U[:, perm]
        end

        nb = max(min(blk, k - nconv, j), 1)

        # Restart if not enough room for nb new vectors
        if j + nb > jmax
            @timing "jdsym_rand_block: restart" begin
                nk = min(jmin, j)
                # QR-orthonormalize: U columns from general eigen are unit-norm but not unitary
                Q_rst = Matrix(qr(U[:, 1:nk]).Q)[:, 1:nk]
                V     = V  * Q_rst
                W     = W  * Q_rst
                SV    = SV * Q_rst
                SW    = SW * Q_rst
                M_proj = Q_rst' * M_proj * Q_rst
                j      = nk
                nb     = max(min(blk, k - nconv, j, jmax - j), 1)
                # Re-diagonalize after restart
                F  = eigen(M_proj)
                ew = real.(F.values)
                U  = F.vectors
                perm = _jdrb_sort_perm(ew, sigma)
                ew = ew[perm]
                U  = U[:, perm]
                disp && @printf("jdsym_rand_block: RESTART j -> %d\n", j)
            end
        end

        # Compute nb Ritz vectors and residuals
        @timing "jdsym_rand_block: residual" begin
            Y_nb     = U[:, 1:nb]
            X_blk    = V * Y_nb
            W_blk    = W * Y_nb
            # Sketched Rayleigh quotients: minimise ‖Θr‖ over real θ.
            # More reliable than real.(M_proj eigenvalues) when M_proj is non-Hermitian.
            SX_rq    = SV * Y_nb
            SW_rq    = SW * Y_nb
            theta_nb = [real(dot(view(SX_rq,:,ib), view(SW_rq,:,ib)) /
                             dot(view(SX_rq,:,ib), view(SX_rq,:,ib))) for ib in 1:nb]
            R_blk    = W_blk - X_blk * Diagonal(theta_nb)
            # Project out converged eigenvectors (double pass, Θ-norm oblique projector)
            if nconv > 0
                for _ in 1:2
                    c = SQschur' * Theta(R_blk)
                    R_blk .-= Qschur * c
                end
            end
        end
        SR_blk = Theta(R_blk)

        nr = [norm(view(SR_blk, :, ib)) for ib in 1:nb]
        history = vcat(history, [maximum(nr) Float64(iter) Float64(nmv)])

        if disp
            @printf("jdsym_rand_block it=%4d nconv=%3d j=%3d nb=%2d |r|=%.3e..%.3e\n",
                    iter, nconv, j, nb, minimum(nr), maximum(nr))
        end

        # Convergence check: consecutive from pair 1 (skip first iteration)
        nconv_new = 0
        if iter > 1
            for ib in 1:nb
                if nr[ib] < tol
                    nconv_new += 1
                else
                    break
                end
            end
        end

        if nconv_new > 0
            # QR-orthonormalize converged Ritz vectors (U cols not unitary for general eigen)
            QY_conv = Matrix(qr(U[:, 1:nconv_new]).Q)[:, 1:nconv_new]
            for ib in 1:nconv_new
                nconv += 1
                v_conv = V * QY_conv[:, ib]
                # Θ-orthogonalize against already-stored converged vectors (numerical safety)
                if nconv > 1
                    for _ in 1:2
                        c = SQschur[:, 1:nconv-1]' * Theta(v_conv)
                        v_conv .-= Qschur[:, 1:nconv-1] * c
                    end
                end
                sv_conv = Theta(v_conv)
                nv = norm(sv_conv)
                nv > 1e-14 && (v_conv ./= nv)
                Qschur  = hcat(Qschur,  v_conv)
                SQschur = hcat(SQschur, Theta(v_conv))
                push!(lambda_conv, theta_nb[ib])
                if disp
                    @printf("  >>> band %d CONVERGED: e=%.10e  |r|=%.3e  at iter %d\n",
                            nconv, theta_nb[ib], nr[ib], iter)
                end
            end

            nconv >= k && break

            # Deflate: project unconverged coefficients against converged ones (removes
            # Qschur contamination in V for non-Hermitian M_proj), then QR-rotate subspace
            if j > nconv_new
                jnew   = j - nconv_new
                U_rest = copy(U[:, nconv_new+1:j])
                for _ in 1:2
                    U_rest = U_rest - QY_conv * (QY_conv' * U_rest)
                end
                Q_def = Matrix(qr(U_rest).Q)[:, 1:jnew]
                V     = V  * Q_def
                W     = W  * Q_def
                SV    = SV * Q_def
                SW    = SW * Q_def
                M_proj = Q_def' * M_proj * Q_def
                j = jnew
            else
                # Subspace exhausted: reinitialize with a single random vector
                v_new = randn(T, n)
                if nconv > 0
                    for _ in 1:2
                        c = SQschur' * Theta(v_new)
                        v_new .-= Qschur * c
                    end
                end
                sv_new = Theta(v_new)
                nv = norm(sv_new)
                nv > 1e-14 && (v_new ./= nv)
                V  = reshape(v_new, n, 1)
                @timing "jdsym_rand_block: matvec" begin
                    W  = reshape(A * V[:, 1], n, 1)
                    nmv += 1
                end
                SV     = Theta(V)
                SW     = Theta(W)
                M_proj = SV' * SW
                j      = 1
            end

            # Reuse unconverged residuals instead of wasting an iteration.
            # Valid because X_blk[:,nconv_new+1:end] = V_new * Q_def' * U[:,nconv_new+1:nb]
            # = V * U[:,nconv_new+1:nb] (since U[:,nconv_new+1:nb] ⊂ span(Q_def))
            nb -= nconv_new
            if nb > 0
                R_blk    = R_blk[:,  nconv_new+1:end]
                SR_blk   = SR_blk[:, nconv_new+1:end]
                X_blk    = X_blk[:,  nconv_new+1:end]
                theta_nb = theta_nb[ nconv_new+1:end]
            else
                continue
            end
        end

        # Apply preconditioner to residuals (correction vectors)
        @timing "jdsym_rand_block: correction" begin
            T_corr = copy(R_blk)
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, X_blk)
                for ib in 1:nb
                    @views T_corr[:, ib] .= M \ T_corr[:, ib]
                end
            end
            # Sketched oblique projection out of converged + active Ritz vectors:
            # (I - Q*(SQ)†*S) where (SQ)† uses QR of the sketched Ritz block
            SX_blk  = Theta(X_blk)
            F_qr    = qr(SX_blk)
            SX_proj = Matrix(F_qr.Q)[:, 1:nb]   # orthonormal basis for span(SX_blk)
            X_proj  = X_blk / F_qr.R             # such that Theta(X_proj) ≈ SX_proj
            Q_proj  = nconv > 0 ? hcat(Qschur, X_proj)  : X_proj
            SQ_proj = nconv > 0 ? hcat(SQschur, SX_proj) : SX_proj
            T_corr  = theta_orth_block_against(T_corr, Q_proj, SQ_proj, Theta, orth_method)
        end

        # Expand subspace: Θ-orthogonalize correction block against V, then normalize
        @timing "jdsym_rand_block: ortho" begin
            T_corr          = theta_orth_block_against(T_corr, V, SV, Theta, orth_method)
            T_corr, ST_corr = theta_orth_block(T_corr, Theta, orth_method)
        end

        if size(T_corr, 2) > 0
            @timing "jdsym_rand_block: matvec" begin
                W_new = A * T_corr
                nmv  += size(T_corr, 2)
            end
            SW_new = Theta(W_new)
            # Incremental update: M_proj grows by nact rows/cols
            @timing "jdsym_rand_block: overlap" begin
                M_proj = [M_proj SV'*SW_new; ST_corr'*SW ST_corr'*SW_new]
            end
            V  = hcat(V,  T_corr)
            SV = hcat(SV, ST_corr)
            W  = hcat(W,  W_new)
            SW = hcat(SW, SW_new)
        end
    end

    disp && @printf("jdsym_rand_block: done. nconv=%d notcnv=%d iter=%d nmv=%d\n",
                    nconv, k - nconv, nit, nmv)

    # Recover accurate eigenpairs via sketched-to-fully refinement
    X_out, lambda_out = if nconv > 0
        sketched_to_fully(A, Qschur)
    else
        zeros(T, n, 0), Float64[]
    end

    # Fill unconverged slots with best available Ritz approximations
    notcnv = k - nconv
    if notcnv > 0 && size(V, 2) > 0
        j_cur  = size(V, 2)
        F_fin  = eigen(M_proj)
        ew_fin = real.(F_fin.values)
        U_fin  = F_fin.vectors
        perm_f = _jdrb_sort_perm(ew_fin, sigma)
        ew_fin = ew_fin[perm_f]
        U_fin  = U_fin[:, perm_f]
        for i in 1:min(notcnv, j_cur)
            X_out      = hcat(X_out, V * U_fin[:, i])
            lambda_out = vcat(lambda_out, [ew_fin[i]])
        end
    end

    return X_out, lambda_out, history
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""Sort permutation for eigenvalues by target `sigma`."""
function _jdrb_sort_perm(ew::AbstractVector, sigma::Symbol)
    if sigma == :LM
        return sortperm(abs.(ew), rev=true)
    elseif sigma == :SM
        return sortperm(abs.(ew))
    elseif sigma == :LR
        return sortperm(ew, rev=true)
    else   # :SR (default)
        return sortperm(ew)
    end
end
