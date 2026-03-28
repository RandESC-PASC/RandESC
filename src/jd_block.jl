using LinearAlgebra
using Printf

"""
    jdsym_block(A; k=5, kwargs...)

Blocked Jacobi-Davidson for symmetric/Hermitian matrices (smallest eigenvalues).
Direct translation of `cjdsym` from Quantum ESPRESSO (Phys. Rev. Research, 2025).

Computes `nblock` correction vectors per iteration for better BLAS-3 utilization.
Uses double-MGS (MGS2) orthogonalization. No sketching.

Algorithm outline per iteration:
  1. Diagonalize projected matrix Hc → (ew, Vc)
  2. Restart subspace if j + nb > kmax  (keep locked + best active Ritz vectors)
  3. Compute nb Ritz vectors and residuals for active pairs (indices nconv+1:nconv+nb)
  4. Check residual-norm convergence; Ritz-rotate subspace to lock converged pairs
  5. Apply preconditioner to residuals (correction vectors)
  6. Expand subspace: MGS2 against V (incl. locked), normalize, update Hc = V'*AV

Locked (converged) eigenvectors occupy V[:,1:nconv] and are included in every Ritz
diagonalization. No separate deflation or projection against locked vectors is needed;
orthogonalization against V[:,1:j] keeps new vectors orthogonal to locked ones.

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenpairs to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance on residual norms
- `maxit=100`: Maximum outer iterations
- `nblock=-1`: Block size (default: `k`)
- `kmax=-1`: Max subspace dimension (default: `2k + 10`)
- `v0=nothing`: Initial vectors (n × ≤k matrix)
- `M=nothing`: Preconditioner (applied as `M \\ r`)
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info

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
                     kmax::Int=-1,
                     v0=nothing,
                     M=nothing,
                     precond_preparator=nothing,
                     disp::Bool=false)

    n = size(A, 1)
    k = min(k, n)

    if n < 1
        return zeros(Float64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    nblock = nblock < 0 ? k : nblock
    kmax   = kmax < 0 ? min(n, 2k + 10) : kmax

    if k > kmax ÷ 2
        error("kmax too small: need kmax > 2k (k=$k, kmax=$kmax)")
    end

    T = isnothing(v0) ? Float64 : eltype(v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    V    = zeros(T, n, kmax)       # search space basis
    W    = zeros(T, n, kmax)       # A * V
    Hc   = zeros(T, kmax, kmax)    # projected Hamiltonian: V' * W
    Vc   = zeros(T, kmax, kmax)    # eigenvectors of projected problem
    ew   = zeros(Float64, kmax)    # eigenvalues of projected problem
    ub   = zeros(T, n, nblock)     # Ritz vectors
    rb   = zeros(T, n, nblock)     # residuals / corrections
    Vtmp = zeros(T, n, kmax)       # temporary for restart / deflation
    Wtmp = zeros(T, n, kmax)       # temporary for restart / deflation

    lambda = zeros(Float64, k)     # converged eigenvalues

    nconv = 0
    nmv   = 0
    history = zeros(Float64, 0, 3)

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(k, n)
    if !isnothing(v0)
        nc = min(size(v0, 2), j)
        @views V[:, 1:nc] .= v0[:, 1:nc]
        if nc < j
            @views V[:, nc+1:j] .= randn(T, n, j - nc)
        end
    else
        @views V[:, 1:j] .= randn(T, n, j)
    end

    @timing "jdsym_block: ortho" _jdb_mgs2!(V, j)

    @timing "jdsym_block: matvec" @views mul!(W[:, 1:j], A, V[:, 1:j])
    nmv += j

    @timing "jdsym_block: overlap" begin
        @views mul!(Hc[1:j, 1:j], V[:, 1:j]', W[:, 1:j])
        _jdb_symmetrize!(Hc, j)
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        # Diagonalize projected EVP
        @timing "jdsym_block: diag" begin
            @views F = eigen(Hermitian(Hc[1:j, 1:j]))
            ew[1:j]      .= F.values
            Vc[1:j, 1:j] .= F.vectors
        end

        nb = max(min(nblock, k - nconv, j - nconv), 1)

        # Restart if not enough room for nb new vectors
        if j + nb > kmax
            @timing "jdsym_block: restart" begin
                # Keep all nconv locked vectors + best (jmin) active Ritz vectors
                jmin = min(k - nconv, j - nconv)
                jnew = nconv + jmin
                @views mul!(Vtmp[:, 1:jnew], V[:, 1:j], Vc[1:j, 1:jnew])
                @views mul!(Wtmp[:, 1:jnew], W[:, 1:j], Vc[1:j, 1:jnew])
                @views V[:, 1:jnew] .= Vtmp[:, 1:jnew]
                @views W[:, 1:jnew] .= Wtmp[:, 1:jnew]
                Hc[1:kmax, 1:kmax] .= zero(T)
                for i in 1:jnew; Hc[i, i] = ew[i]; end
                j = jnew
                nb = max(min(nblock, k - nconv, j - nconv, kmax - j), 1)
                # Re-diagonalize after restart
                @views F = eigen(Hermitian(Hc[1:j, 1:j]))
                ew[1:j]      .= F.values
                Vc[1:j, 1:j] .= F.vectors
                disp && @printf("cjdsym: RESTART j -> %d\n", j)
            end
        end

        # Guard: if no active vectors remain (j == nconv), seed a new one
        if j <= nconv
            if j < kmax
                j += 1
                @views V[:, j] .= randn(T, n)
                for _ in 1:2
                    @views c = V[:, 1:j-1]' * V[:, j]
                    @views V[:, j] .-= V[:, 1:j-1] * c
                end
                nv = norm(view(V, :, j))
                nv < 1e-14 && (j -= 1; continue)
                @views V[:, j] ./= nv
                @views mul!(W[:, j], A, V[:, j])
                nmv += 1
                @views Hc[1:j, j] .= V[:, 1:j]' * W[:, j]
                @views Hc[j, 1:j-1] .= conj.(Hc[1:j-1, j])
                Hc[j, j] = real(Hc[j, j])
            end
            continue
        end

        @timing "jdsym_block: residual" begin
            # Ritz vectors for active pairs: indices nconv+1 : nconv+nb
            @views mul!(ub[:, 1:nb], V[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            # H * active Ritz vectors
            @views mul!(rb[:, 1:nb], W[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            # Residuals: rb -= ew[nconv+ib] * ub  (rb = (A - theta*I) * u)
            for ib in 1:nb
                @views rb[:, ib] .-= ew[nconv+ib] .* ub[:, ib]
            end
            # No explicit projection against locked vectors:
            # orthogonalization in _jdb_expand! against V[:,1:j] handles it.
        end

        rnorms = [norm(view(rb, :, ib)) for ib in 1:nb]
        history = vcat(history, [maximum(rnorms) Float64(iter) Float64(nmv)])

        if disp
            @printf("cjdsym it=%4d nconv=%3d j=%3d nb=%2d |r|=%.3e..%.3e\n",
                    iter, nconv, j, nb, minimum(rnorms), maximum(rnorms))
        end

        # Convergence check: consecutive from pair 1
        nconv_new = 0
        # if iter > 1
            for ib in 1:nb
                if rnorms[ib] < tol
                    nconv_new += 1
                else
                    break
                end
            end
        # end

        if nconv_new > 0
            if disp
                for ib in 1:nconv_new
                    @printf("  >>> band %d CONVERGED: e=%.10e  |r|=%.3e  at iter %d\n",
                            nconv + ib, ew[nconv + ib], rnorms[ib], iter)
                end
            end
            for ib in 1:nconv_new
                lambda[nconv + ib] = ew[nconv + ib]
            end

            # Ritz rotation: bring V/W/Hc to the Ritz basis so that
            # V[:,1:nconv+nconv_new] become the converged eigenvectors.
            # Locked vectors stay at the front; j is unchanged.
            @views mul!(Vtmp[:, 1:j], V[:, 1:j], Vc[1:j, 1:j])
            @views mul!(Wtmp[:, 1:j], W[:, 1:j], Vc[1:j, 1:j])
            @views V[:, 1:j] .= Vtmp[:, 1:j]
            @views W[:, 1:j] .= Wtmp[:, 1:j]
            Hc[1:kmax, 1:kmax] .= zero(T)
            for i in 1:j; Hc[i, i] = ew[i]; end

            nconv += nconv_new
            nconv >= k && break

            # Reuse unconverged residuals instead of wasting an iteration
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

        # Apply preconditioner (correction step)
        @timing "jdsym_block: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, view(ub, :, 1:nb))
                for ib in 1:nb
                    @views rb[:, ib] .= M \ rb[:, ib]
                end
            end
            # No explicit projection against locked vectors needed:
            # _jdb_expand! orthogonalizes against all of V[:,1:j] including locked.
        end

        # Expand subspace
        nact = _jdb_expand!(V, W, Hc, rb, j, nb, kmax, A)
        nmv += nact
        j += nact
    end

    disp && @printf("cjdsym: done. nconv=%d notcnv=%d iter=%d nmv=%d\n",
                    nconv, k - nconv, jd_iter, nmv)

    # Construct output eigenvectors.
    # V[:,1:nconv] are the converged eigenvectors (placed there by Ritz rotations).
    # Fill any unconverged slots with the best active Ritz approximations.
    X_out = zeros(T, n, k)
    @views X_out[:, 1:nconv] .= V[:, 1:nconv]

    notcnv = k - nconv
    if notcnv > 0 && j > nconv
        @views Fout = eigen(Hermitian(Hc[1:j, 1:j]))
        nfill = min(notcnv, j - nconv)
        @views mul!(X_out[:, nconv+1:nconv+nfill], V[:, 1:j], Fout.vectors[1:j, nconv+1:nconv+nfill])
        for i in 1:nfill
            lambda[nconv + i] = Fout.values[nconv + i]
        end
        for i in nfill+1:notcnv
            lambda[nconv + i] = Fout.values[nconv + min(nfill, j - nconv)]
        end
    end

    return X_out, lambda, history
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""Double-pass MGS orthonormalization of V[:,1:m] in-place."""
function _jdb_mgs2!(V::AbstractMatrix, m::Int)
    for i in 1:m
        if i > 1
            for _ in 1:2
                @views c = V[:, 1:i-1]' * V[:, i]
                @views V[:, i] .-= V[:, 1:i-1] * c
            end
        end
        nv = norm(view(V, :, i))
        nv > 1e-14 && (@views V[:, i] ./= nv)
    end
end


"""Symmetrize H[1:m,1:m]: force real diagonal, H[i,j] = conj(H[j,i])."""
function _jdb_symmetrize!(H::AbstractMatrix, m::Int)
    for i in 1:m
        H[i, i] = real(H[i, i])
        for j in i+1:m
            H[i, j] = conj(H[j, i])
        end
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
                      j::Int, nb::Int, kmax::Int, A)
    nact = 0

    @timing "jdsym_block: ortho" begin
        # Phase 1: orthogonalize all nb corrections against existing V[:,1:j]
        for _ in 1:2
            @views c = V[:, 1:j]' * rb[:, 1:nb]
            @views rb[:, 1:nb] .-= V[:, 1:j] * c
        end

        # Phase 2: sequential orthogonalization among the nb new vectors
        for ib in 1:nb
            j + nact >= kmax && break   # subspace full

            if nact > 0
                for _ in 1:2
                    @views c = V[:, j+1:j+nact]' * rb[:, ib]
                    @views rb[:, ib] .-= V[:, j+1:j+nact] * c
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
