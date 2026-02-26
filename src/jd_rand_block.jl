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

    # Pre-allocate converged eigenpair storage (no hcat growth)
    Qschur  = zeros(T, n, k)
    SQschur = zeros(T, s, k)
    nconv = 0

    nmv   = 0
    nit   = 0
    # Pre-allocate history (no vcat growth); truncated to nit rows at return
    history = zeros(Float64, maxit, 3)

    # ── Pre-allocate double buffers for V/SV/W/SW (zero-allocation rotations/expansions) ──
    # Active subspace lives in columns 1:j of the _a buffers.
    # Rotations use mul! into _b then swap; expansions write directly into _a[:, j+1:j+nb].
    V_a  = zeros(T, n, jmax);  V_b  = zeros(T, n, jmax)
    SV_a = zeros(T, s, jmax);  SV_b = zeros(T, s, jmax)
    W_a  = zeros(T, n, jmax);  W_b  = zeros(T, n, jmax)
    SW_a = zeros(T, s, jmax);  SW_b = zeros(T, s, jmax)

    # ── Initialize subspace ───────────────────────────────────────────────
    v0_init = isnothing(v0) ? randn(T, n, blk) : v0
    if size(v0_init, 2) < blk
        v0_init = hcat(v0_init, randn(T, n, blk - size(v0_init, 2)))
    end
    V_init, SV_init = init_space_sketched(n, v0_init, Theta,
                                          view(Qschur, :, 1:0), view(SQschur, :, 1:0),
                                          T, orth_method, jmax)
    j = size(V_init, 2)
    V_a[:, 1:j]  .= V_init
    SV_a[:, 1:j] .= SV_init

    @timing "jdsym_rand_block: matvec" begin
        mul!(view(W_a, :, 1:j), A, view(V_a, :, 1:j))
        nmv += j
    end
    @timing "jdsym_rand_block: sketch" begin
        SW_a[:, 1:j] .= Theta(view(W_a, :, 1:j))
    end

    V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
    W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)

    # M_proj = (ΘV)'*(ΘAV); general complex matrix (approximately Hermitian for Hermitian A)
    @timing "jdsym_rand_block: overlap" M_proj = SV' * SW

    # ── Main loop ──────────────────────────────────────────────────────────
    for iter in 1:maxit
        nit = iter

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
                # Zero-allocation rotation via double-buffer swap
                mul!(view(V_b,  :, 1:nk), V,  Q_rst)
                mul!(view(SV_b, :, 1:nk), SV, Q_rst)
                mul!(view(W_b,  :, 1:nk), W,  Q_rst)
                mul!(view(SW_b, :, 1:nk), SW, Q_rst)
                V_a, V_b   = V_b,  V_a
                SV_a, SV_b = SV_b, SV_a
                W_a, W_b   = W_b,  W_a
                SW_a, SW_b = SW_b, SW_a
                j  = nk
                V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
                W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)
                nb = max(min(blk, k - nconv, j, jmax - j), 1)
                M_proj = Q_rst' * M_proj * Q_rst
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
                Qsc  = view(Qschur,  :, 1:nconv)
                SQsc = view(SQschur, :, 1:nconv)
                for _ in 1:2
                    c = SQsc' * Theta(R_blk)
                    R_blk .-= Qsc * c
                end
            end
        end
        @timing "jdsym_rand_block: sketch" SR_blk = Theta(R_blk)

        nr = [norm(view(SR_blk, :, ib)) for ib in 1:nb]
        history[nit, 1] = maximum(nr)
        history[nit, 2] = Float64(iter)
        history[nit, 3] = Float64(nmv)

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

        @timing "jdsym_rand_block: converge" if nconv_new > 0
            # QR-orthonormalize converged Ritz vectors (U cols not unitary for general eigen)
            QY_conv = Matrix(qr(U[:, 1:nconv_new]).Q)[:, 1:nconv_new]
            for ib in 1:nconv_new
                nconv += 1
                v_conv = V * QY_conv[:, ib]
                # Θ-orthogonalize against already-stored converged vectors (numerical safety)
                if nconv > 1
                    Qsc  = view(Qschur,  :, 1:nconv-1)
                    SQsc = view(SQschur, :, 1:nconv-1)
                    for _ in 1:2
                        c = SQsc' * Theta(v_conv)
                        v_conv .-= Qsc * c
                    end
                end
                sv_conv = Theta(v_conv)
                nv = norm(sv_conv)
                nv > 1e-14 && (v_conv ./= nv; sv_conv ./= nv)
                Qschur[:,  nconv] .= v_conv     # in-place: no hcat allocation
                SQschur[:, nconv] .= Theta(v_conv)
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
                # Zero-allocation rotation via double-buffer swap
                mul!(view(V_b,  :, 1:jnew), V,  Q_def)
                mul!(view(SV_b, :, 1:jnew), SV, Q_def)
                mul!(view(W_b,  :, 1:jnew), W,  Q_def)
                mul!(view(SW_b, :, 1:jnew), SW, Q_def)
                V_a, V_b   = V_b,  V_a
                SV_a, SV_b = SV_b, SV_a
                W_a, W_b   = W_b,  W_a
                SW_a, SW_b = SW_b, SW_a
                j  = jnew
                V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
                W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)
                M_proj = Q_def' * M_proj * Q_def
            else
                # Subspace exhausted: reinitialize with a single random vector
                v_new = randn(T, n)
                if nconv > 0
                    Qsc  = view(Qschur,  :, 1:nconv)
                    SQsc = view(SQschur, :, 1:nconv)
                    for _ in 1:2
                        c = SQsc' * Theta(v_new)
                        v_new .-= Qsc * c
                    end
                end
                sv_new = Theta(v_new)
                nv = norm(sv_new)
                nv > 1e-14 && (v_new ./= nv)
                V_a[:, 1] .= v_new
                @timing "jdsym_rand_block: matvec" begin
                    mul!(view(W_a, :, 1:1), A, view(V_a, :, 1:1))
                    nmv += 1
                end
                @timing "jdsym_rand_block: sketch" begin
                    SV_a[:, 1] .= Theta(view(V_a, :, 1:1))
                    SW_a[:, 1] .= Theta(view(W_a, :, 1:1))
                end
                j  = 1
                V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
                W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)
                M_proj = SV' * SW
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
            # (I - Q*(SQ)†*S) where (SQ)† uses QR of the sketched Ritz block.
            # Use thin Q (avoid materialising the full s×s unitary matrix).
            SX_blk  = Theta(X_blk)
            F_qr    = qr(SX_blk)
            SX_proj = F_qr.Q * Matrix{T}(I, size(SX_blk, 1), nb)  # thin Q, s×nb
            X_proj  = X_blk / F_qr.R             # such that Theta(X_proj) ≈ SX_proj
            if nconv > 0
                Qsc  = view(Qschur,  :, 1:nconv)
                SQsc = view(SQschur, :, 1:nconv)
                Q_proj  = hcat(Qsc,  X_proj)
                SQ_proj = hcat(SQsc, SX_proj)
            else
                Q_proj  = X_proj
                SQ_proj = SX_proj
            end
            T_corr  = theta_orth_block_against(T_corr, Q_proj, SQ_proj, Theta, orth_method)
        end

        # Expand subspace: Θ-orthogonalize correction block against V, then normalize
        @timing "jdsym_rand_block: ortho" begin
            T_corr          = theta_orth_block_against(T_corr, V, SV, Theta, orth_method)
            T_corr, ST_corr = theta_orth_block(T_corr, Theta, orth_method)
        end

        if size(T_corr, 2) > 0
            nb_new = size(T_corr, 2)
            @timing "jdsym_rand_block: matvec" begin
                mul!(view(W_a, :, j+1:j+nb_new), A, T_corr)
                nmv += nb_new
            end
            @timing "jdsym_rand_block: sketch" begin
                SW_a[:, j+1:j+nb_new] .= Theta(view(W_a, :, j+1:j+nb_new))
                SV_a[:, j+1:j+nb_new] .= ST_corr
            end
            # Incremental M_proj update (before changing j so SV/SW still point to old block)
            @timing "jdsym_rand_block: overlap" begin
                SW_new = view(SW_a, :, j+1:j+nb_new)
                M_proj = [M_proj      SV'*SW_new;
                          ST_corr'*SW ST_corr'*SW_new]
            end
            # Write T_corr into V buffer and update views (zero allocation)
            @timing "jdsym_rand_block: expand" begin
                V_a[:, j+1:j+nb_new] .= T_corr
                j += nb_new
                V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
                W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)
            end
        end
    end

    disp && @printf("jdsym_rand_block: done. nconv=%d notcnv=%d iter=%d nmv=%d\n",
                    nconv, k - nconv, nit, nmv)

    # Recover accurate eigenpairs via sketched-to-fully refinement
    X_out, lambda_out = if nconv > 0
        sketched_to_fully(A, view(Qschur, :, 1:nconv))
    else
        zeros(T, n, 0), Float64[]
    end

    # Fill unconverged slots with best available Ritz approximations
    notcnv = k - nconv
    if notcnv > 0 && j > 0
        F_fin  = eigen(M_proj)
        ew_fin = real.(F_fin.values)
        U_fin  = F_fin.vectors
        perm_f = _jdrb_sort_perm(ew_fin, sigma)
        ew_fin = ew_fin[perm_f]
        U_fin  = U_fin[:, perm_f]
        for i in 1:min(notcnv, j)
            X_out      = hcat(X_out, V * U_fin[:, i])
            lambda_out = vcat(lambda_out, [ew_fin[i]])
        end
    end

    return X_out, lambda_out, history[1:nit, :]
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
