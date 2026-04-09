using LinearAlgebra
using Printf

"""
    jdsym_block(A; k=5, kwargs...)

Davidson eigensolver for symmetric/Hermitian matrices (smallest eigenvalues).
Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. nbuff=2 buffer pairs are kept beyond k. Search space restarts
to jmin=2*(k+nbuff) vectors when full, and grows up to kmax=4*(k+nbuff).

Arguments:
- A: Hermitian matrix or operator
- k: number of eigenpairs (default 5)
- tol: residual norm convergence threshold (default 1e-8)
- maxit: max iterations (default 100)
- v0: initial vectors (n x m matrix, optional)
- M: preconditioner, applied as M \\ r (optional)
- precond_preparator: callback f(M, X) to refresh preconditioner (optional)
- disp: print iteration info

Returns (X, lambda, history) where X is n x k, lambda is length k,
and history is nit x 3 with columns [max_rnorm, iter, nmv].
"""
@views @timing "jdsym_block" function jdsym_block(A; k::Int=5,
                     tol::Float64=1e-8,
                     maxit::Int=100,
                     v0=nothing,
                     M=nothing,
                     precond_preparator=nothing,
                     disp::Bool=false)

    n = size(A, 1)
    k = min(k, n)
    nbuff = 2 # buffer of unrequested vector that are added to search space
    kb = min(k + nbuff, n) # block size
    jmin = min(n, 2 * (k + nbuff)) # search space size after restart
    kmax = min(n, 4kb) # maximal search space dimension

    two_pass_orth = false # if true, orthogonalization is more accurate (and more expensive)

    T = isnothing(v0) ? Float64 : eltype(v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jdsym_block: allocation" begin
        V      = zeros(T, n, kmax)    # search space basis (includes soft-locked)
        W      = zeros(T, n, kmax)    # A * V
        rb     = zeros(T, n, kb)      # residuals / corrections
        ub     = zeros(T, n, kb)      # Ritz vectors for active pairs
        buffer = zeros(T, n, jmin)    # Pre-allocated work array
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Hc: projected Hamiltonian, j x j T Hermitian matrix
    # Vc: Ritz vercors of the projected problem, j x j T matrix
    # ew: Ritz values of the projected problem, length j Real vector

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    if !isnothing(v0)
        nc = min(size(v0, 2), j)
        V[:, 1:nc] .= v0[:, 1:nc]
        if nc < j
            V[:, nc+1:j] .= randn(T, n, j - nc)
        end
    else
        V[:, 1:j] .= randn(T, n, j)
    end

    @timing "jdsym_block: ortho" _jdb_mgs2!(V, j, two_pass_orth)
    @timing "jdsym_block: matvec" mul!(W[:, 1:j], A, V[:, 1:j])
    nmv += j
    @timing "jdsym_block: overlap" Hc = Hermitian(V[:, 1:j]' * W[:, 1:j])

    # Initial full subspace diagonalization
    @timing "jdsym_block: diag" begin
        ew, Vc = eigen(Hc)
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep kb Ritz vectors (soft-locked stay in V)
        if j + nb >= kmax
            @timing "jdsym_block: restart" begin
                mul!(buffer[:, 1:jmin], V[:, 1:j], Vc[1:j, 1:jmin])
                V[:, 1:jmin] .= buffer[:, 1:jmin]
                mul!(buffer[:, 1:jmin], W[:, 1:j], Vc[1:j, 1:jmin])
                W[:, 1:jmin] .= buffer[:, 1:jmin]
                Hc = Hermitian(Diagonal(T.(ew[1:jmin])))
                j = jmin
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                # Because Hc is diagonal, eigenvalues and eigevectors are trivial
                ew = ew[1:jmin]
                Vc = I(jmin)
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (blocked BLAS-3)
        @timing "jdsym_block: residual" begin
            mul!(ub[:, 1:nb], V[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            mul!(rb[:, 1:nb], W[:, 1:j], Vc[1:j, nconv+1:nconv+nb])
            rb[:, 1:nb] .-= ub[:, 1:nb] .* ew[nconv+1:nconv+nb]'
            rnorms = columnwise_norms(rb[:, 1:nb])
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
        history[hist_row, :] .= (maximum(rnorms[1:nb_target]), iter, nmv)

        if disp
            @printf("jdsym_block it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                    iter, nconv, k, j, nb,
                    minimum(rnorms[1:nb_target]), maximum(rnorms[1:nb_target]))
        end

        # Lock newly converged pairs (Ritz vectors stay in V via soft-locking)
        if nconv_new > 0
            @timing "jdsym_block: lock" begin
                nconv += nconv_new
                nconv >= k && break
                # Shift remaining active residuals to front
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
        @timing "jdsym_block: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, ub[:, 1:nb])
                ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
                rb[:, 1:nb] .= ub[:, 1:nb]
            end
        end

        # Expand subspace, new expanded Hc comes out
        Hc, nact = _jdb_expand!(V, W, Hc, rb, j, nb, kmax, A, two_pass_orth)
        nmv += nact
        j += nact

        # Diagonalize over full subspace (soft-locked pairs stay in V)
        @timing "jdsym_block: diag" begin
            ew, Vc = eigen(Hc)
        end
    end

    disp && @printf("jdsym_block: done. nconv=%d/%d  iter=%d  nmv=%d\n", nconv, k, jd_iter, nmv)

    if nconv < k
        @warn "jdsym_block did not converge within $maxit iterations."
    end

    # Use final Ritz values — consistent with the eigenvectors computed below
    lambda = ew[1:min(k, j)]

    # Compute eigenvectors from current Ritz vectors
    X = V[:, 1:j] * Vc[1:j, 1:k]

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""Double-pass MGS orthonormalization of V[:,1:m] in-place."""
function _jdb_mgs2!(V::AbstractMatrix, m::Int, two_pass::Bool=true)
    for i in 1:m
        if i > 1
            vi = view(V, :, i)
            Vp = view(V, :, 1:i-1)
            ci = Vp' * vi
            mul!(vi, Vp, ci, -1, 1)
            if two_pass
                mul!(ci, Vp', vi)
                mul!(vi, Vp, ci, -1, 1)
            end
        end
        nv = norm(view(V, :, i))
        nv > 1e-14 && (view(V, :, i) ./= nv)
    end
end



"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Phase 1 (BLAS-3): orthogonalize all nb vectors at once against V[:,1:j].
Phase 2 (sequential): orthogonalize among the nb new vectors, normalize, accept.
Then compute A * new columns and returns the expanded projected Hamiltonian Hexp.

Returns the expanded projected Hamiltonian `Hexp` and the number of accepted vectors `nact`.
"""
@views function _jdb_expand!(V::AbstractMatrix, W::AbstractMatrix,
                      Hc::AbstractMatrix, rb::AbstractMatrix,
                      j::Int, nb::Int, kmax::Int, A,
                      two_pass::Bool=true)
    nact = 0

    @timing "jdsym_block: ortho" begin
        # Phase 1: orthogonalize all nb corrections against existing V[:,1:j]
        let Vj = view(V, :, 1:j), rbv = view(rb, :, 1:nb)
            c1v = Vj' *rbv
            mul!(rbv, Vj, c1v, -1, 1)
            if two_pass
                mul!(c1v, Vj', rbv)
                mul!(rbv, Vj, c1v, -1, 1)
            end
        end

        # Phase 2: sequential orthogonalization among the nb new vectors
        #      can we go block by block?
        for ib in 1:nb
            j + nact >= kmax && break   # subspace full

            if nact > 0
                let Vnew = view(V, :, j+1:j+nact), rbib = view(rb, :, ib)
                    c2v = Vnew' * rbib
                    mul!(rbib, Vnew, c2v, -1, 1)
                    if two_pass
                        mul!(c2v, Vnew', rbib)
                        mul!(rbib, Vnew, c2v, -1, 1)
                    end
                end
            end

            nv = norm(view(rb, :, ib))
            nv < 1e-14 && continue      # numerically zero, skip

            rb[:, ib] ./= nv
            nact += 1
            V[:, j + nact] .= rb[:, ib]
        end
    end

    nact == 0 && return copy(Hc), 0

    # Compute A * new basis vectors
    @timing "jdsym_block: matvec" mul!(W[:, j+1:j+nact], A, V[:, j+1:j+nact])

    # Update projected Hamiltonian (full column block, BLAS-3)
    @timing "jdsym_block: overlap" begin
        Hexp = similar(Hc, j+nact, j+nact)
        Hexp[1:j, 1:j] .= Hc
        mul!(Hexp[1:j+nact, j+1:j+nact], V[:, 1:j+nact]', W[:, j+1:j+nact])
        Hexp = Hermitian(Hexp)  # Hermitian ==> no need for explicit symmetrization
    end

    return Hexp, nact
end

# Calculate the norms of the columns of an array
function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end
