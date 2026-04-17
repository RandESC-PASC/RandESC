using LinearAlgebra
using Printf


"""
    jd_sketched(A, v0; k=5, kwargs...)

Sketched Jacobi-Davidson eigensolver for symmetric/Hermitian matrices.
Structure mirrors `jd` exactly; the only differences are:
  - V is kept Θ-orthonormal (sketched orthogonalization) instead of standard orthonormal
  - The projected matrix is Mc = (ΘV)'(ΘAV) instead of Hc = V'AV
  - Convergence is measured in Θ-norm: ‖Θr‖ instead of ‖r‖
  - Rayleigh quotients are computed via the sketched inner product

Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. nbuff=2 buffer pairs are kept beyond k. Search space restarts
to jmin=2*(k+nbuff) vectors when full, and grows up to jmax=4*(k+nbuff).

# Arguments
- `A`: Symmetric/Hermitian matrix or operator
- `v0`: Initial vectors (n x m matrix)
- `k`: Number of eigenpairs (default 5)

# Keyword Arguments
- `tol=1e-8`: Residual Θ-norm convergence threshold
- `maxit=200`: Maximum iterations
- `M=nothing`: Preconditioner, applied as `M \\ r`
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info
- `sketch_type="sparsestack"`: Sketch operator type (see sketch.jl)
- `sketch_size=-1`: Sketch dimension s (default: `max(5*jmax, 5*k)`)
- `orth_method=:rgs`: Θ-orthogonalization method (`:rgs`, `:rcgs`, `:rcgs2`)

# Returns
`(X, lambda, history)` where X is nxk, lambda is length k,
and history is nitx3 with columns [max_rnorm, iter, nmv].
"""
@views @timing "jd_sketched" function jd_sketched(A, v0::AbstractArray;
                          k::Int=5,
                          tol::Float64=1e-8,
                          maxit::Int=200,
                          M=nothing,
                          precond_preparator=nothing,
                          disp::Bool=false,
                          sketch_type::String="sparsesign",
                          sketch_size::Int=-1,
                          orth_method::Symbol=:rgs)

    n = size(A, 1)
    k = min(k, n)
    nbuff = 2                         # buffer of unrequested pairs added to search space
    kb   = min(k + nbuff, n)          # block size
    jmin = min(n, 2 * (k + nbuff))    # search space size after restart
    jmax = min(n, 4 * kb)             # maximal search space dimension
    s    = sketch_size < 0 ? max(5 * jmax, 5 * k) : sketch_size

    if disp
        println("Dimension in jd_sketched")
        println("n: $n, k: $k, s: $s")
    end


    T = complex(eltype(v0))

    Theta = sketch(n, s, sketch_type; template=v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jd_sketched: allocation" begin
        # Active subspace: columns 1:j live in the the following buffers.
        V  = similar(v0, n, jmax)
        SV = similar(v0, s, jmax)
        W  = similar(v0, n, jmax)
        SW = similar(v0, s, jmax)
        # Work buffers for various allocation free operations:
        n_buffer = similar(v0, n, jmax)
        s_buffer = similar(v0, s, jmax)
        # Ritz vectors and residuals for active pairs (like ub/rb in jdsym_block)
        # ub is also reused as T_corr_buf during expand (never live at the same time)
        ub      = similar(v0, n, kb)
        rb      = similar(v0, n, kb)
        # Sketched Ritz vectors for Rayleigh quotient; reused as ST_corr_buf/SV_scratch
        # during expand (never live at the same time as the Rayleigh quotient computation)
        SX_rq   = similar(v0, s, kb)   # reused as ST_corr_buf in expand
        SW_rq   = similar(v0, s, kb)   # reused as SV_scratch in expand
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Mc: projected matrix, j x j, Mc = (ΘV)'(ΘAV)
    # theta_b: Rayleigh quotients for active Ritz pairs, length kb
    # U: Ritz vectors of the projected problem, j x j
    # ew: Ritz values of the projected problem, length j, Real vector

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    @timing "jd_sketched: init" begin
        nc = min(size(v0, 2), j)
        V[:, 1:nc] .= v0[:, 1:nc]
        if nc < j
            randn!(TaskLocalRNG(), V[:, nc+1:j])   # in-place, no temp allocation
        end
        # Θ-orthonormalize in-place into V_a/SV_a using pre-allocated scratch.
        # V_a[:,1:j] is both input and output — safe because column i is consumed
        # into v_work before column ncols≤i is overwritten.
        # SW_rq (s×kb) used as SV_buf scratch; not yet populated.
        j = theta_orth_block!(V[:, 1:j], SV[:, 1:j], V[:, 1:j],
                              Theta, orth_method,
                              SW_rq)
    end
    @timing "jd_sketched: matvec" begin
        mul!(W[:, 1:j], A, V[:, 1:j])
        nmv += j
    end
    @timing "jd_sketched: sketch" begin
        mul!(SW[:, 1:j], Theta, W[:, 1:j])
    end
    # Mc = (ΘV)'(ΘAV), replaces Hc = V'AV
    @timing "jd_sketched: overlap" begin
        Mc = SV[:, 1:j]' * SW[:, 1:j]
    end

    # Initial Mc diagonalization (general eigen; only approx. Hermitian for sketching)
    @timing "jd_sketched: diag" begin
        F    = eigen(Mc)
        ew   = real.(F.values)
        U    = F.vectors
        perm = sortperm(ew)
        ew   = copy(ew[perm])     # Note: use copy to avoid accidental non-contiguous views
        U    = copy(U[:, perm])
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep jmin Ritz vectors (soft-locked pairs stay in V)
        if j + nb >= jmax
            @timing "jd_sketched: restart" begin
                nk    = min(jmin, j)
                # QR-orthonormalize: U columns from general eigen not guaranteed unitary
                # Use work buffers to avoid extra allocations
                Q_rst = oftype(V, qr(U[:, 1:nk]).Q)
                mul!(n_buffer[:, 1:nk], V[:, 1:j],  Q_rst)
                V[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SV[:, 1:j], Q_rst)
                SV[:, 1:nk] .= s_buffer[:, 1:nk]
                mul!(n_buffer[:, 1:nk], W[:, 1:j],  Q_rst)
                W[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SW[:, 1:j], Q_rst)
                SW[:, 1:nk] .= s_buffer[:, 1:nk]
                j  = nk
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                # Rotate Mc to the new basis (like jd_block's Vc'*Hc*Vc, but without the
                # unitary simplification — Q_rst is only approx. unitary for non-Hermitian Mc)
                Mc = Q_rst' * Mc * Q_rst   # nk×nk
                # Re-diagonalize after restart (like jd_block's eigen(Hermitian(Hc[1:j,1:j])))
                F_rst = eigen(Mc)
                ew    = real.(F_rst.values)
                U     = F_rst.vectors
                perm  = sortperm(ew)
                ew    = copy(ew[perm]);  U = copy(U[:, perm])
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (BLAS-3, like jd)
        @timing "jd_sketched: residual" begin
            Y_nb = view(U, :, nconv+1:nconv+nb)
            mul!(ub[:, 1:nb], V[:, 1:j], Y_nb)
            mul!(rb[:, 1:nb], W[:, 1:j], Y_nb)
            # Sketched Rayleigh quotients: more reliable than real.(ew) for non-Hermitian Mc
            mul!(SX_rq[:, 1:nb], SV[:, 1:j], Y_nb)
            mul!(SW_rq[:, 1:nb], SW[:, 1:j], Y_nb)
            theta_b = real(columnwise_dots(SX_rq[:, 1:nb], SW_rq[:, 1:nb]) ./
                           columnwise_dots(SX_rq[:, 1:nb], SX_rq[:, 1:nb]))
            # r = AX - X*diag(theta), in-place (replaces jd_block's rb .-= ub .* ew')
            @views rb[:, 1:nb] .-= ub[:, 1:nb] .* theta_b[1:nb]'
        end

        # Sketch residuals for Θ-norm computation — reuse sv_work to avoid per-column alloc
        @timing "jd_sketched: sketch" begin
            mul!(SW_rq[:, 1:nb], Theta, rb[:, 1:nb])   # reuse SW_rq as tmp buffer
            rnorms = Array(columnwise_norms(SW_rq[:, 1:nb]))
        end

        # Convergence check on active target pairs (skip first iteration, like jd)
        @timing "jd_sketched: check" begin
            nb_target = min(k - nconv, nb)

            hist_row += 1
            history[hist_row, :] .= (maximum(rnorms[1:nb_target]), iter, nmv)

            if disp
                @printf("jd_sketched it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                        iter, nconv, k, j, nb,
                        minimum(rnorms[1:nb_target]), maximum(rnorms[1:nb_target]))
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
            @timing "jd_sketched: lock" begin
                nconv += nconv_new
                nconv >= k && break
                # Shift remaining active residuals/Ritz-vectors to front (like jd)
                nb -= nconv_new
                if nb > 0
                    rb[:, 1:nb] .= rb[:, nconv_new + 1:nconv_new + nb]
                    ub[:, 1:nb] .= ub[:, nconv_new + 1:nconv_new + nb]
                else
                    continue
                end
            end
        end

        # Apply preconditioner
        @timing "jd_sketched: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, ub[:, 1:nb])
                ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
                rb[:, 1:nb] .= ub[:, 1:nb]
            end
        end

        # Expand subspace: Θ-orthogonalize corrections against V, return new expanded Mc.
        # Mirrors _jdb_expand! but with sketched orthogonalization and Mc instead of Hc.
        # ub=T_corr_buf, SX_rq=ST_corr_buf, SW_rq=SV_scratch — merged buffers, not live here
        Mc, nact = _jdrb_expand!(V, SV, W, SW, Mc, rb, j, nb, jmax, A, Theta, orth_method,
                                 ub, SX_rq, SW_rq)
        nmv += nact
        j   += nact

        # Diagonalize expanded Mc (general eigen; only approx. Hermitian for sketching)
        @timing "jd_sketched: diag" begin
            F    = eigen(Mc)
            ew   = real.(F.values)
            U    = F.vectors
            perm = sortperm(ew)
            ew   = copy(ew[perm])
            U    = copy(U[:, perm])
        end
    end

    disp && @printf("jd_sketched: done. nconv=%d/%d  iter=%d  nmv=%d\n", nconv, k, jd_iter, nmv)

    if nconv < k
        @warn "jd_sketched did not converge within $maxit iterations."
    end

    # Final Ritz extraction and post-processing.
    # V is Θ-orthonormal (not standard orthonormal) and U from general eigen is non-unitary,
    # so V*U[:,1:k] is not orthogonal.
    # Perform one final, full ritz step to get the best approximation from the converged search space.
    @timing "jd_sketched: finalize" begin
        F_fin  = eigen(Mc)
        ew_fin = real.(F_fin.values)
        U_fin  = F_fin.vectors
        perm_f = sortperm(ew_fin)
        U_fin  = copy(U_fin[:, perm_f[1:k]])             # j×k, small
        mul!(n_buffer[:, 1:k], V[:, 1:j], U_fin)
        V[:, 1:k] .= n_buffer[:, 1:k]
        mul!(n_buffer[:, 1:k], W[:, 1:j], U_fin)
        W[:, 1:k] .= n_buffer[:, 1:k]
        Xritz = view(V, :, 1:k)
        Writz = view(W, :, 1:k)
        # Cholesky-orthonormalize Xritz in-place: Xritz → Xritz / R, Writz → Writz / R
        K = Xritz' * Xritz                         # k×k, small
        R = cholesky(Hermitian(K)).U
        rdiv!(Xritz, R)   # Xritz = Vn, in-place
        rdiv!(Writz, R)   # Writz = AVn, in-place
        # Small k×k projected eigenproblem
        Fk = eigen(Hermitian(Xritz' * Writz))
        # Final output: only this n×k allocation is unavoidable.
        # Convert back to input type: real inputs give real outputs (imaginary parts are ~0).
        X_full = Xritz * Fk.vectors
        lambda = Fk.values
        perm_s = sortperm(lambda)
        X_sorted = X_full[:, perm_s]
        if eltype(v0) <: Real
            # Eigenvectors computed in complex arithmetic carry an arbitrary scalar phase
            # e^(iφ) per column. Remove it so that real.() discards only imaginary noise.
            # maybe the phase can already be removed in the projected eigenvalue problem that is
            # solved a couple of lines above.
            for j in axes(X_sorted, 2)
                col = view(X_sorted, :, j)
                _, idx = findmax(abs, col)
                col ./= col[idx] / abs(col[idx])
            end
            X = real.(X_sorted)
        else
            X = copy(X_sorted)
        end
        lambda = copy(lambda[perm_s])
    end

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Mirrors `_jdb_expand!` from jd, but uses Θ-orthogonalization instead of MGS.

Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j].
Phase 2: Θ-orthogonalize among the nb new vectors; normalize; accept.
Then compute A * new columns, sketch, and expands Mc[:,j+1:j+nact] and Mc[j+1:j+nact,:].

Returns the expanded Mc and the number of accepted vectors `nact`.
"""
@views function _jdrb_expand!(V, SV, W, SW, Mc, rb, j, nb, jmax, A, Theta, orth_method,
                       T_corr_buf, ST_corr_buf, SV_scratch)
    # T_corr_buf  = ub   (n×kb, reused from Ritz-vector buffer — not live during expand)
    # ST_corr_buf = SX_rq (s×kb, reused from sketched-Ritz buffer — not live during expand)
    # SV_scratch  = SW_rq (s×kb, reused from sketched-AV buffer  — not live during expand)

    # Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j] — in-place, no alloc
    @timing "jd_sketched: ortho" begin
        rbb = view(rb, :, 1:nb)
        theta_orth_block_against!(rbb, V[:, 1:j], SV[:, 1:j], Theta, orth_method, SV_scratch)
    end

    # Phase 2: Θ-orthonormalize among the nb corrections into pre-allocated buffers
    @timing "jd_sketched: ortho" begin
        nact = theta_orth_block!(T_corr_buf, ST_corr_buf, rb[:, 1:nb], Theta,
                                 orth_method, SV_scratch)
    end

    nact == 0 && return Mc, 0
    nact = min(nact, jmax - j)
    nact == 0 && return Mc, 0

    T_corr  = view(T_corr_buf,  :, 1:nact)
    ST_corr = view(ST_corr_buf, :, 1:nact)

    # Compute A * new basis vectors and their sketches
    @timing "jd_sketched: matvec" mul!(W[:, j+1:j+nact], A, T_corr)
    @timing "jd_sketched: sketch" begin
        mul!(SW[:, j+1:j+nact], Theta, W[:, j+1:j+nact])
        SV[:, j+1:j+nact] .= ST_corr
    end

    # Expand Mc (mirrors jd's Hc column+row update, BLAS-3).
    @timing "jd_sketched: overlap" begin
        SW_new = view(SW, :, j+1:j+nact)
        SW_old = view(SW, :, 1:j)
        Mexp = similar(Mc, j+nact, j+nact)
        Mexp[1:j, 1:j] .= Mc
        mul!(Mexp[1:j,        j+1:j+nact], SV[:, 1:j]',      SW_new)
        mul!(Mexp[j+1:j+nact, 1:j],        ST_corr', SW_old)
        mul!(Mexp[j+1:j+nact, j+1:j+nact], ST_corr', SW_new)
    end

    # Write T_corr into V buffer
    @timing "jd_sketched: expand" begin
        V[:, j+1:j+nact] .= T_corr
    end

    return Mexp, nact
end