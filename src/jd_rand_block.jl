using LinearAlgebra
using Printf


"""
    jdsym_rand_block(A; k=5, kwargs...)

Sketched Jacobi-Davidson eigensolver for symmetric/Hermitian matrices.
Structure mirrors `jdsym_block` exactly; the only differences are:
  - V is kept Θ-orthonormal (sketched orthogonalization) instead of standard orthonormal
  - The projected matrix is Mc = (ΘV)'(ΘAV) instead of Hc = V'AV
  - Convergence is measured in Θ-norm: ‖Θr‖ instead of ‖r‖
  - Rayleigh quotients are computed via the sketched inner product

Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. nbuff=2 buffer pairs are kept beyond k. Search space restarts
to jmin=2*(k+nbuff) vectors when full, and grows up to jmax=4*(k+nbuff).

# Arguments
- `A`: Symmetric/Hermitian matrix or operator
- `k`: Number of eigenpairs (default 5)

# Keyword Arguments
- `tol=1e-8`: Residual Θ-norm convergence threshold
- `maxit=200`: Maximum iterations
- `v0=nothing`: Initial vectors (n x m matrix, optional)
- `sigma=:SR`: Target: `:SR` smallest real, `:LR` largest real,
               `:SM` smallest magnitude, `:LM` largest magnitude
- `M=nothing`: Preconditioner, applied as `M \\ r`
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info
- `sketch_type="sparsestack"`: Sketch operator type (see sketch.jl)
- `sketch_size=-1`: Sketch dimension s (default: `max(4*jmax, 4*k)`)
- `orth_method=:rgs`: Θ-orthogonalization method (`:rgs`, `:rcgs`, `:rcgs2`)

# Returns
`(X, lambda, history)` where X is nxk, lambda is length k,
and history is nitx3 with columns [max_rnorm, iter, nmv].
"""
@timing "jdsym_rand_block" function jdsym_rand_block(A; k::Int=5,
                          tol::Float64=1e-8,
                          maxit::Int=200,
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
    nbuff = 2                         # buffer of unrequested pairs added to search space
    kb   = min(k + nbuff, n)          # block size
    jmin = min(n, 2 * (k + nbuff))    # search space size after restart
    jmax = min(n, 4 * kb)             # maximal search space dimension
    s    = sketch_size < 0 ? max(6jmax, 6k) : sketch_size
    # s = 1000
    # sketch_type = "complex_gaussian"

    println("n: $n, k: $k, s: $s")


    T = isnothing(v0) ? ComplexF64 : eltype(v0)

    Theta = sketch(n, s, sketch_type)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jdsym_rand_block: allocation" begin
        # Active subspace: columns 1:j live in the _a buffers.
        # Restart/rotation uses mul! into _b then pointer swap (zero allocation).
        V_a  = zeros(T, n, jmax);  V_b  = zeros(T, n, jmax)
        SV_a = zeros(T, s, jmax);  SV_b = zeros(T, s, jmax)
        W_a  = zeros(T, n, jmax);  W_b  = zeros(T, n, jmax)
        SW_a = zeros(T, s, jmax);  SW_b = zeros(T, s, jmax)
        # Projected matrix Mc[1:j, 1:j] = (ΘV)'(ΘAV) — pre-allocated at full size
        # like Hc in jdsym_block; updated incrementally in _jdrb_expand!
        Mc      = zeros(T, jmax, jmax)
        # Ritz vectors and residuals for active pairs (like ub/rb in jdsym_block)
        # ub is also reused as T_corr_buf during expand (never live at the same time)
        ub      = zeros(T, n, kb)
        rb      = zeros(T, n, kb)
        lambda  = zeros(Float64, k)
        rnorms  = zeros(Float64, kb)
        theta_b = zeros(Float64, kb)   # Ritz values for active pairs
        # Sketched Ritz vectors for Rayleigh quotient; reused as ST_corr_buf/SV_scratch
        # during expand (never live at the same time as the Rayleigh quotient computation)
        SX_rq   = zeros(T, s, kb)   # reused as ST_corr_buf in expand
        SW_rq   = zeros(T, s, kb)   # reused as SV_scratch in expand
        # H_buf: projection coefficients in theta_orth_block_against! (jmax×kb)
        H_buf   = zeros(T, jmax, kb)
        v_work  = zeros(T, n)
        sv_work = zeros(T, s)
        h_work  = zeros(T, kb)
    end

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    @timing "jdsym_rand_block: init" begin
        if !isnothing(v0)
            nc = min(size(v0, 2), j)
            @views V_a[:, 1:nc] .= v0[:, 1:nc]
            if nc < j
                randn!(view(V_a, :, nc+1:j))   # in-place, no temp allocation
            end
        else
            randn!(view(V_a, :, 1:j))           # in-place, no temp allocation
        end
        # Θ-orthonormalize in-place into V_a/SV_a using pre-allocated scratch.
        # V_a[:,1:j] is both input and output — safe because column i is consumed
        # into v_work before column ncols≤i is overwritten.
        # SW_rq (s×kb) used as SV_buf scratch; not yet populated.
        j = theta_orth_block!(view(V_a, :, 1:j), view(SV_a, :, 1:j),
                              view(V_a, :, 1:j), Theta, orth_method,
                              SW_rq, v_work, sv_work, h_work)
    end
    @timing "jdsym_rand_block: matvec" begin
        mul!(view(W_a, :, 1:j), A, view(V_a, :, 1:j))
        nmv += j
    end
    @timing "jdsym_rand_block: sketch" begin
        mul!(view(SW_a, :, 1:j), Theta, view(W_a, :, 1:j))
    end
    # Mc[1:j, 1:j] = (ΘV)'(ΘAV), replaces Hc = V'AV
    @timing "jdsym_rand_block: overlap" begin
        mul!(view(Mc, 1:j, 1:j), view(SV_a, :, 1:j)', view(SW_a, :, 1:j))
    end

    # pre-declare ew so it is accessible for the post-loop finalize
    ew = zeros(Float64, j)

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        V  = view(V_a,  :, 1:j);  SV = view(SV_a, :, 1:j)
        W  = view(W_a,  :, 1:j);  SW = view(SW_a, :, 1:j)

        # Diagonalize Mc[1:j,1:j] (general eigen; only approx. Hermitian for sketching)
        @timing "jdsym_rand_block: diag" begin
            F    = eigen(view(Mc, 1:j, 1:j))
            ew   = real.(F.values)
            U    = F.vectors
            perm = _jdrb_sort_perm(ew, sigma)
            ew   = ew[perm]
            U    = U[:, perm]
        end

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep jmin Ritz vectors (soft-locked pairs stay in V)
        if j + nb >= jmax
            @timing "jdsym_rand_block: restart" begin
                nk    = min(jmin, j)
                j_old = j
                # QR-orthonormalize: U columns from general eigen not guaranteed unitary
                Q_rst = Matrix(qr(U[:, 1:nk]).Q)[:, 1:nk]
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
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                # Rotate Mc to the new basis (like jd_block's Vc'*Hc*Vc, but without the
                # unitary simplification — Q_rst is only approx. unitary for non-Hermitian Mc)
                Mc_tmp = Q_rst' * view(Mc, 1:j_old, 1:j_old) * Q_rst   # nk×nk
                Mc[1:jmax, 1:jmax] .= zero(T)
                Mc[1:nk, 1:nk] .= Mc_tmp
                # Re-diagonalize after restart (like jd_block's eigen(Hermitian(Hc[1:j,1:j])))
                F_rst = eigen(view(Mc, 1:j, 1:j))
                ew    = real.(F_rst.values)
                U     = F_rst.vectors
                perm  = _jdrb_sort_perm(ew, sigma)
                ew    = ew[perm];  U = U[:, perm]
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (BLAS-3, like jdsym_block)
        @timing "jdsym_rand_block: residual" begin
            Y_nb = U[:, nconv+1:nconv+nb]
            mul!(view(ub, :, 1:nb), V, Y_nb)
            mul!(view(rb, :, 1:nb), W, Y_nb)
            # Sketched Rayleigh quotients: more reliable than real.(ew) for non-Hermitian Mc
            mul!(view(SX_rq, :, 1:nb), SV, Y_nb)
            mul!(view(SW_rq, :, 1:nb), SW, Y_nb)
            for ib in 1:nb
                theta_b[ib] = real(dot(view(SX_rq, :, ib), view(SW_rq, :, ib)) /
                                   dot(view(SX_rq, :, ib), view(SX_rq, :, ib)))
            end
            # r = AX - X*diag(theta), in-place (replaces jd_block's rb .-= ub .* ew')
            @views rb[:, 1:nb] .-= ub[:, 1:nb] .* theta_b[1:nb]'
        end

        # Sketch residuals for Θ-norm computation — reuse sv_work to avoid per-column alloc
        @timing "jdsym_rand_block: sketch" begin
            for ib in 1:nb
                mul!(sv_work, Theta, view(rb, :, ib))
                rnorms[ib] = norm(sv_work)
            end
        end

        # Convergence check on active target pairs (skip first iteration, like jdsym_block)
        @timing "jdsym_rand_block: check" begin
            nb_target = min(k - nconv, nb)

            hist_row += 1
            history[hist_row, :] .= (maximum(view(rnorms, 1:nb_target)), iter, nmv)

            if disp
                @printf("jdsym_rand_block it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                        iter, nconv, k, j, nb,
                        minimum(view(rnorms, 1:nb_target)), maximum(view(rnorms, 1:nb_target)))
            end

            nconv_new = 0
            if iter > 1
                for ib in 1:nb_target
                    rnorms[ib] < tol ? nconv_new += 1 : break
                end
            end
        end

        # Lock newly converged pairs (Ritz vectors stay in V via soft-locking)
        if nconv_new > 0
            @timing "jdsym_rand_block: lock" begin
                for _ in 1:nconv_new
                    nconv += 1
                    lambda[nconv] = ew[nconv]
                end
            end
            nconv >= k && break
            # Shift remaining active residuals/Ritz-vectors to front (like jdsym_block)
            nb -= nconv_new
            if nb > 0
                for ib in 1:nb
                    @views rb[:, ib] .= rb[:, nconv_new + ib]
                    @views ub[:, ib] .= ub[:, nconv_new + ib]
                end
            else
                continue
            end
        end

        # Apply preconditioner
        @timing "jdsym_rand_block: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, view(ub, :, 1:nb))
                @views ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
                @views rb[:, 1:nb] .= ub[:, 1:nb]
            end
        end

        # Expand subspace: Θ-orthogonalize corrections against V, update Mc in-place.
        # Mirrors _jdb_expand! but with sketched orthogonalization and Mc instead of Hc.
        # ub=T_corr_buf, SX_rq=ST_corr_buf, SW_rq=SV_scratch — merged buffers, not live here
        nact = _jdrb_expand!(V_a, SV_a, W_a, SW_a, Mc, rb, j, nb, jmax, A, Theta, orth_method,
                             ub, SX_rq, SW_rq, H_buf, v_work, sv_work, h_work)
        nmv += nact
        j   += nact
    end

    disp && @printf("jdsym_rand_block: done. nconv=%d/%d  iter=%d  nmv=%d\n", nconv, k, jd_iter, nmv)

    if nconv < k
        @warn "jdsym_rand_block did not converge within $maxit iterations."
    end

    # Final Ritz extraction and post-processing.
    # V is Θ-orthonormal (not standard orthonormal) and U from general eigen is non-unitary,
    # so V*U[:,1:k] is not orthogonal. We inline sketched_to_fully, reusing V_b[:,1:k] and
    # W_b[:,1:k] (idle after main loop) for X_ritz and W_ritz, then dividing in-place with
    # rdiv! to avoid 4 additional n×k allocations. Only the final n×k output must be allocated.
    @timing "jdsym_rand_block: finalize" begin
        F_fin  = eigen(view(Mc, 1:j, 1:j))
        ew_fin = real.(F_fin.values)
        U_fin  = F_fin.vectors
        perm_f = _jdrb_sort_perm(ew_fin, sigma)
        U_fin  = U_fin[:, perm_f[1:k]]             # j×k, small
        Xritz  = view(V_b, :, 1:k)                 # reuse V_b as scratch for Ritz vectors
        Writz  = view(W_b, :, 1:k)                 # reuse W_b as scratch for A*Ritz
        mul!(Xritz, view(V_a, :, 1:j), U_fin)
        mul!(Writz, view(W_a, :, 1:j), U_fin)
        # Cholesky-orthonormalize Xritz in-place: Xritz → Xritz / R, Writz → Writz / R
        K = Xritz' * Xritz                         # k×k, small
        R = cholesky(Hermitian(K)).U
        rdiv!(Xritz, R)   # Xritz = Vn, in-place
        rdiv!(Writz, R)   # Writz = AVn, in-place
        # Small k×k projected eigenproblem
        Fk = eigen(Hermitian(Xritz' * Writz))
        # Final output: only this n×k allocation is unavoidable
        X      = Xritz * Fk.vectors
        lambda = Fk.values
        perm_s = _jdrb_sort_perm(lambda, sigma)
        X      = X[:, perm_s]
        lambda = lambda[perm_s]
    end

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Mirrors `_jdb_expand!` from jdsym_block, but uses Θ-orthogonalization instead of MGS,
and updates Mc (pre-allocated jmax×jmax, replaces Hc) in-place incrementally.

Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j].
Phase 2: Θ-orthogonalize among the nb new vectors; normalize; accept.
Then compute A * new columns, sketch, and update Mc[:,j+1:j+nact] and Mc[j+1:j+nact,:].

Returns the number of accepted vectors `nact`.
"""
function _jdrb_expand!(V_a, SV_a, W_a, SW_a, Mc, rb, j, nb, jmax, A, Theta, orth_method,
                       T_corr_buf, ST_corr_buf, SV_scratch, H_buf, v_work, sv_work, h_work)
    # T_corr_buf  = ub   (n×kb, reused from Ritz-vector buffer — not live during expand)
    # ST_corr_buf = SX_rq (s×kb, reused from sketched-Ritz buffer — not live during expand)
    # SV_scratch  = SW_rq (s×kb, reused from sketched-AV buffer  — not live during expand)
    V  = view(V_a,  :, 1:j)
    SV = view(SV_a, :, 1:j)

    # Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j] — in-place, no alloc
    @timing "jdsym_rand_block: ortho" begin
        rbb = view(rb, :, 1:nb)
        theta_orth_block_against!(rbb, V, SV, Theta, orth_method, SV_scratch, H_buf)
    end

    # Phase 2: Θ-orthonormalize among the nb corrections into pre-allocated buffers
    @timing "jdsym_rand_block: ortho" begin
        nact = theta_orth_block!(T_corr_buf, ST_corr_buf, view(rb, :, 1:nb), Theta,
                                 orth_method, SV_scratch, v_work, sv_work, h_work)
    end

    nact == 0 && return 0
    nact = min(nact, jmax - j)
    nact == 0 && return 0

    T_corr  = view(T_corr_buf,  :, 1:nact)
    ST_corr = view(ST_corr_buf, :, 1:nact)

    # Compute A * new basis vectors and their sketches
    @timing "jdsym_rand_block: matvec" mul!(view(W_a, :, j+1:j+nact), A, T_corr)
    @timing "jdsym_rand_block: sketch" begin
        mul!(view(SW_a, :, j+1:j+nact), Theta, view(W_a, :, j+1:j+nact))
        SV_a[:, j+1:j+nact] .= ST_corr
    end

    # Update Mc in-place (mirrors jdsym_block's Hc column+row update, BLAS-3).
    @timing "jdsym_rand_block: overlap" begin
        SW_new = view(SW_a, :, j+1:j+nact)
        SW_old = view(SW_a, :, 1:j)
        mul!(view(Mc, 1:j,          j+1:j+nact), SV',      SW_new)
        mul!(view(Mc, j+1:j+nact,   1:j),        ST_corr', SW_old)
        mul!(view(Mc, j+1:j+nact,   j+1:j+nact), ST_corr', SW_new)
    end

    # Write T_corr into V buffer
    @timing "jdsym_rand_block: expand" begin
        V_a[:, j+1:j+nact] .= T_corr
    end

    return nact
end


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
