using LinearAlgebra
using Printf

"""
    jdsym_block(A; k=5, kwargs...)

Davidson eigensolver for symmetric/Hermitian matrices (smallest eigenvalues).

Soft-locking: converged Ritz vectors stay in V so the full subspace is always used
in the Ritz diagonalization. Corrections are generated only for active (unconverged)
pairs. _jdb_expand! orthogonalizes against all of V, implicitly keeping corrections
in the orthogonal complement of the soft-locked subspace.

Algorithm outline per iteration:
  1. Diagonalize projected matrix Hc → (ew, Vc)  [full subspace incl. soft-locked]
  2. Restart if j + nb > kmax  (keep kb = k+nbuff Ritz vectors, incl. soft-locked)
  3. Compute residuals for active pairs nconv+1..nconv+nb (blocked BLAS-3)
  4. Check consecutive convergence; update monotone nconv
  5. Apply preconditioner to active residuals
  6. Expand subspace: MGS2 against V, normalize, update Hc = V'*AV

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenpairs to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance on residual norms
- `maxit=100`: Maximum outer iterations
- `nblock=-1`: Unused (kept for API compatibility)
- `nbuff=0`: Extra buffer pairs; corrections generated for k-nconv+nbuff active pairs
- `kmax=-1`: Max subspace dimension (default: `2*(k+nbuff) + k + 10`; the extra `k` accounts for soft-locked vectors occupying slots)
- `v0=nothing`: Initial vectors (n × ≤kb matrix)
- `M=nothing`: Preconditioner (applied as `M \\ r`)
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info
- `debug=false`: Print detailed debug info

# Returns
`(X, lambda, history)` where
- `X` (n × k): eigenvectors
- `lambda` (k,): eigenvalues in ascending order
- `history` (nit × 3): columns are [max_rnorm, iter, nmv]
"""
@timing "jdsym_block" function jdsym_block(A; k::Int=5,
                     tol::Float64=1e-8,
                     maxit::Int=100,
                     nblock::Int=-1,
                     nbuff::Int=2,
                     kmax::Int=-1,
                     v0=nothing,
                     M=nothing,
                     precond_preparator=nothing,
                     disp::Bool=false,
                     debug::Bool=false)

    n = size(A, 1)
    k = min(k, n)
    kb = min(k + nbuff, n)

    if n < 1
        return zeros(Float64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    kmax = kmax < 0 ? min(n, 4kb) : kmax

    if kb > kmax ÷ 2
        error("kmax too small: need kmax > 2*(k+nbuff) (kb=$kb, kmax=$kmax)")
    end

    T = isnothing(v0) ? Float64 : eltype(v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jdsym_block: allocation" begin
        V      = zeros(T, n, kmax)    # search space basis (includes soft-locked)
        W      = zeros(T, n, kmax)    # A * V
        Hc     = zeros(T, kmax, kmax) # projected Hamiltonian
        Vc     = zeros(T, kmax, kmax) # Ritz vectors of projected problem
        ew     = zeros(Float64, kmax) # Ritz values
        ub     = zeros(T, n, kb)      # Ritz vectors for active pairs
        rb     = zeros(T, n, kb)      # residuals / corrections
        Vtmp   = zeros(T, n, k + kb)  # scratch for restart (nconv + kb ≤ k + kb)
        Wtmp   = zeros(T, n, k + kb)  # scratch for restart
        lambda = zeros(Float64, k)
        rnorms = zeros(Float64, kb)
        c_mgs  = zeros(T, kmax)
        c_exp1 = zeros(T, kmax, kb)
        c_exp2 = zeros(T, kb)
    end

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    if !isnothing(v0)
        nc = min(size(v0, 2), j)
        @views V[:, 1:nc] .= v0[:, 1:nc]
        if nc < j
            @views V[:, nc+1:j] .= randn(T, n, j - nc)
        end
    else
        @views V[:, 1:j] .= randn(T, n, j)
    end

    @timing "jdsym_block: ortho" _jdb_mgs2!(V, j, c_mgs)
    @timing "jdsym_block: matvec" @views mul!(W[:, 1:j], A, V[:, 1:j])
    nmv += j
    @timing "jdsym_block: overlap" @views mul!(Hc[1:j, 1:j], V[:, 1:j]', W[:, 1:j])

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        # Diagonalize over full subspace (soft-locked pairs stay in V)
        @timing "jdsym_block: diag" begin
            @views F = eigen(Hermitian(Hc[1:j, 1:j]))
            ew[1:j]      .= F.values
            Vc[1:j, 1:j] .= F.vectors
        end

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep kb Ritz vectors (soft-locked stay in V)
        if j + nb > kmax
            @timing "jdsym_block: restart" begin
                jnew = min(nconv + kb, j)   # keep nconv soft-locked + kb active Ritz vectors
                @views mul!(Vtmp[:, 1:jnew], V[:, 1:j], Vc[1:j, 1:jnew])
                @views mul!(Wtmp[:, 1:jnew], W[:, 1:j], Vc[1:j, 1:jnew])
                @views V[:, 1:jnew] .= Vtmp[:, 1:jnew]
                @views W[:, 1:jnew] .= Wtmp[:, 1:jnew]
                Hc[1:kmax, 1:kmax] .= zero(T)
                for i in 1:jnew; Hc[i, i] = ew[i]; end
                j = jnew
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                @views F = eigen(Hermitian(Hc[1:j, 1:j]))
                ew[1:j]      .= F.values
                Vc[1:j, 1:j] .= F.vectors
                (disp || debug) && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (blocked BLAS-3)
        @timing "jdsym_block: residual" begin
            @views mul!(ub[:, 1:nb], V[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            @views mul!(rb[:, 1:nb], W[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            @views rb[:, 1:nb] .-= ub[:, 1:nb] .* ew[nconv+1:nconv+nb]'
            for ib in 1:nb; rnorms[ib] = norm(view(rb, :, ib)); end
        end

        # Consecutive convergence check (target pairs only, skip first iteration)
        nb_target = min(k - nconv, nb)
        nconv_new = 0
        if iter > 1
            for ib in 1:nb_target
                rnorms[ib] < tol ? nconv_new += 1 : break
            end
        end

        hist_row += 1
        history[hist_row, :] .= (maximum(view(rnorms, 1:nb_target)), iter, nmv)

        if disp
            @printf("jdsym_block it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                    iter, nconv, k, j, nb,
                    minimum(view(rnorms, 1:nb_target)), maximum(view(rnorms, 1:nb_target)))
        end
        if debug
            ew_str = nb_target <= 6 ?
                join([@sprintf("%.4e", ew[nconv+i]) for i in 1:nb_target], " ") :
                @sprintf("%.4e..%.4e", ew[nconv+1], ew[nconv+nb_target])
            @printf("DEBUG it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e  ew=[%s]\n",
                    iter, nconv, k, j, nb,
                    minimum(view(rnorms, 1:nb_target)), maximum(view(rnorms, 1:nb_target)), ew_str)
        end

        # Lock newly converged pairs (Ritz vectors stay in V via soft-locking)
        if nconv_new > 0
            @timing "jdsym_block: lock" begin
                for _ in 1:nconv_new
                    nconv += 1
                    lambda[nconv] = ew[nconv]
                end
            end
            nconv >= k && break
            # Shift remaining active residuals to front
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
        @timing "jdsym_block: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, view(ub, :, 1:nb))
                @views ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
                @views rb[:, 1:nb] .= ub[:, 1:nb]
            end
        end

        # Expand subspace
        nact = _jdb_expand!(V, W, Hc, rb, j, nb, kmax, A, c_exp1, c_exp2)
        nmv += nact
        j += nact
        debug && nact < nb && @printf("DEBUG           expand: only %d/%d vectors accepted (j=%d)\n", nact, nb, j)
    end

    disp && @printf("jdsym_block: done. nconv=%d/%d  iter=%d  nmv=%d\n", nconv, k, jd_iter, nmv)

    if nconv < k
        @warn "jdsym_block did not converge within $maxit iterations."
        # Fill unconverged slots with best available Ritz approximations
        nb_fill = min(k - nconv, max(j - nconv, 0))
        for i in 1:nb_fill
            lambda[nconv + i] = ew[nconv + i]
        end
    end

    # Compute eigenvectors from current Ritz vectors
    X = zeros(T, n, k)
    @views mul!(X, V[:, 1:j], Vc[1:j, 1:k])

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""Double-pass MGS orthonormalization of V[:,1:m] in-place. `c` is a preallocated buffer of length ≥ m-1."""
function _jdb_mgs2!(V::AbstractMatrix, m::Int, c::AbstractVector)
    for i in 1:m
        if i > 1
            ci = view(c, 1:i-1)
            vi = view(V, :, i)
            Vp = view(V, :, 1:i-1)
            mul!(ci, Vp', vi)
            mul!(vi, Vp, ci, -1, 1)
            mul!(ci, Vp', vi)
            mul!(vi, Vp, ci, -1, 1)
        end
        nv = norm(view(V, :, i))
        nv > 1e-14 && (view(V, :, i) ./= nv)
    end
end



"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Phase 1 (BLAS-3): orthogonalize all nb vectors at once against V[:,1:j].
Phase 2 (sequential): orthogonalize among the nb new vectors, normalize, accept.
Then compute A * new columns and update the projected Hamiltonian Hc.

Returns the number of accepted vectors `nact`.
"""
function _jdb_expand!(V::AbstractMatrix, W::AbstractMatrix,
                      Hc::AbstractMatrix, rb::AbstractMatrix,
                      j::Int, nb::Int, kmax::Int, A,
                      c1::AbstractMatrix, c2::AbstractVector)
    nact = 0

    @timing "jdsym_block: ortho" begin
        # Phase 1: orthogonalize all nb corrections against existing V[:,1:j]
        let Vj = view(V, :, 1:j), rbv = view(rb, :, 1:nb), c1v = view(c1, 1:j, 1:nb)
            mul!(c1v, Vj', rbv)
            mul!(rbv, Vj, c1v, -1, 1)
            mul!(c1v, Vj', rbv)
            mul!(rbv, Vj, c1v, -1, 1)
        end

        # Phase 2: sequential orthogonalization among the nb new vectors
        for ib in 1:nb
            j + nact >= kmax && break   # subspace full

            if nact > 0
                let Vnew = view(V, :, j+1:j+nact), rbib = view(rb, :, ib), c2v = view(c2, 1:nact)
                    mul!(c2v, Vnew', rbib)
                    mul!(rbib, Vnew, c2v, -1, 1)
                    mul!(c2v, Vnew', rbib)
                    mul!(rbib, Vnew, c2v, -1, 1)
                end
            end

            nv = norm(view(rb, :, ib))
            nv < 1e-14 && continue      # numerically zero, skip

            @views rb[:, ib] ./= nv
            nact += 1
            @views V[:, j + nact] .= rb[:, ib]
        end
    end

    nact == 0 && return 0

    # Compute A * new basis vectors
    @timing "jdsym_block: matvec" @views mul!(W[:, j+1:j+nact], A, V[:, j+1:j+nact])

    # Update projected Hamiltonian (full column block, BLAS-3)
    @timing "jdsym_block: overlap" begin
        @views mul!(Hc[1:j+nact, j+1:j+nact], V[:, 1:j+nact]', W[:, j+1:j+nact])
        for ib in 1:nact
            jj = j + ib
            Hc[jj, jj] = real(Hc[jj, jj])
            for ii in 1:jj-1
                Hc[jj, ii] = conj(Hc[ii, jj])
            end
        end
    end

    return nact
end
