using LinearAlgebra
using Printf

"""
    jd(A, v0; k=5, kwargs...)

Davidson eigensolver for symmetric/Hermitian matrices (smallest eigenvalues).
Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. nbuff=2 buffer pairs are kept beyond k. Search space restarts
to jmin=2*(k+nbuff) vectors when full, and grows up to kmax=4*(k+nbuff).

Arguments:
- A: Hermitian matrix or operator
- v0: initial vectors (n x m matrix)
- k: number of eigenpairs (default 5)
- tol: residual norm convergence threshold (default 1e-8)
- maxit: max iterations (default 100)
- M: preconditioner, applied as M \\ r (optional)
- precond_preparator: callback f(M, X) to refresh preconditioner (optional)
- disp: print iteration info
- orth_method: orthogonalization method (:mgs, :mgs2, :qr) mgs better on cpu, qr on gpu

Returns (X, lambda, history) where X is n x k, lambda is length k,
and history is nit x 3 with columns [max_rnorm, iter, nmv].
"""
@views @timing "jd" function jd(A, v0::AbstractArray;
                     k::Int=5,
                     tol::Float64=1e-8,
                     maxit::Int=100,
                     M=nothing,
                     precond_preparator=nothing,
                     disp::Bool=false,
                     orth_method::Symbol=:mgs)

    n = size(A, 1)
    k = min(k, n)
    nbuff = 2 # buffer of unrequested vector that are added to search space
    kb = min(k + nbuff, n) # block size
    jmin = min(n, 2 * (k + nbuff)) # search space size after restart
    kmax = min(n, 4kb) # maximal search space dimension

    T = eltype(v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    # Note: inherit architecture from input v0 (CPU, NVIDIA GPU, AMD GPU, etc.).
    #       The resulting code is vendor agnostic.
    @timing "jd: allocation" begin
        V      = similar(v0, n, kmax)    # search space basis (includes soft-locked)
        W      = similar(v0, n, kmax)    # A * V
        
        softlocked = zeros(Bool, k) # false if not softlocked, true if softlocked
        hardlocked = zeros(Bool, k) # false if not hardlocked, true if hardlocked
        highest_soft_locked_ind = 0 # index of highest soft locked eigenvalue
        highest_hard_locked_ind = 0 # index of highest hard locked eigenvalue
        lambda_hl  = zeros(real(T), k)   # eigenvalues of hard-locked pairs

        rb     = similar(v0, n, kb)      # residuals / corrections
        ub     = similar(v0, n, kb)      # Ritz vectors for active pairs
        buffer = similar(v0, n, jmin)    # Pre-allocated work array
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Hc: projected Hamiltonian, j x j T Hermitian matrix
    # Vc: Ritz vectors of the projected problem, j x j T matrix
    # ew: Ritz values of the projected problem, length j Real vector

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    nc = min(size(v0, 2), j)
    V[:, 1:nc] .= v0[:, 1:nc]
    if nc < j
        randn!(TaskLocalRNG(), V[:, nc+1:j])
    end

    @timing "jd: ortho" _ortho!(V, j, orth_method, buffer)
    @timing "jd: matvec" mul!(W[:, 1:j], A, V[:, 1:j])
    nmv += j
    @timing "jd: overlap" Hc = Hermitian(V[:, 1:j]' * W[:, 1:j])

    # Initial full subspace diagonalization
    @timing "jd: diag" begin
        ew, Vc = eigen(Hc)
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    nhl_sum = 0
    for iter in 1:maxit
        jd_iter = iter

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nhl = highest_hard_locked_ind
        nhl_sum += nhl
        nconv_r = nconv - nhl   # soft-locked count in reduced space V[:,nhl+1:j]
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep kb Ritz vectors (soft-locked stay in V)
        if j + nb >= kmax
            @timing "jd: restart" begin
                jmin_r = jmin - nhl
                mul!(buffer[:, 1:jmin_r], V[:, nhl+1:j], Vc[:, 1:jmin_r])
                V[:, nhl+1:nhl+jmin_r] .= buffer[:, 1:jmin_r]
                mul!(buffer[:, 1:jmin_r], W[:, nhl+1:j], Vc[:, 1:jmin_r])
                W[:, nhl+1:nhl+jmin_r] .= buffer[:, 1:jmin_r]
                Hc = Hermitian(Diagonal(T.(ew[1:jmin_r])))
                j = nhl + jmin_r
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                # Because Hc is diagonal, eigenvalues and eigenvectors are trivial
                ew = ew[1:jmin_r]
                Vc = similar(V, jmin_r, jmin_r)  # identity matrix. mul! with GPU arrays
                fill!(Vc, zero(T))               # do not like I, so explicit matrix
                Vc[diagind(Vc)] .= one(T)
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (blocked BLAS-3)
        @timing "jd: residual" begin
            mul!(ub[:, 1:nb], V[:, nhl+1:j], Vc[:, nconv_r+1:nconv_r+nb])
            mul!(rb[:, 1:nb], W[:, nhl+1:j], Vc[:, nconv_r+1:nconv_r+nb])
            rb[:, 1:nb] .-= ub[:, 1:nb] .* ew[nconv_r+1:nconv_r+nb]'
            rnorms = columnwise_norms(rb[:, 1:nb])
        end

        # Consecutive convergence check (target pairs only, skip first iteration)
        nb_target = min(k - nconv, nb)
        nconv_new = 0
        if iter > 1
            first_notconv = findfirst(rnorms[1:nb_target] .>= tol)
            nconv_new = isnothing(first_notconv) ? nb_target : first_notconv - 1

        end

        hist_row += 1
        history[hist_row, :] .= (maximum(rnorms[1:nb_target]), iter, nmv)

        if disp
            @printf("jd it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                    iter, nconv, k, j, nb,
                    minimum(rnorms[1:nb_target]), maximum(rnorms[1:nb_target]))
        end

        # Lock newly converged pairs
        if nconv_new > 0
            @timing "jd: lock" begin
                # soft lock part: 
                highest_soft_locked_ind += nconv_new
                softlocked[nconv+1:nconv+nconv_new] .= true
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
                # hard lock part: gap criterion in reduced space (ew indexed from 1 after trimming)
                nconv_r = nconv - nhl
                nhardlocknew = findfirst(ew[nconv_r] .- ew[1:nconv_r] .< .2)
                if isnothing(nhardlocknew)
                    nhardlocknew = 0
                else
                    nhardlocknew -= 1
                end
                if nhardlocknew > 0
                    hardlocked[highest_hard_locked_ind+1:highest_hard_locked_ind+nhardlocknew] .= true
                    highest_hard_locked_ind += nhardlocknew
                    # Rotate V[:,nhl+1:j] to Ritz basis so hard-locked Ritz vectors
                    # land in V[:,nhl+1:nhl_new] and remaining in V[:,nhl_new+1:j]
                    @timing "jd: hardlock" begin
                        jr_old = j - nhl
                        lambda_hl[nhl+1:highest_hard_locked_ind] .= ew[1:nhardlocknew]
                        Vtmp = V[:, nhl+1:j] * Vc[:, 1:jr_old]
                        V[:, nhl+1:j] .= Vtmp
                        Wtmp = W[:, nhl+1:j] * Vc[:, 1:jr_old]
                        W[:, nhl+1:j] .= Wtmp
                        nhl = highest_hard_locked_ind
                        jr = j - nhl
                        Hc = Hermitian(Diagonal(T.(ew[nhardlocknew+1:nhardlocknew+jr])))
                        ew = ew[nhardlocknew+1:nhardlocknew+jr]
                        Vc = similar(V, jr, jr)
                        fill!(Vc, zero(T))
                        Vc[diagind(Vc)] .= one(T)
                        nconv_r = nconv - nhl
                    end
                end

            end
        end

        # Apply preconditioner
        @timing "jd: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, ub[:, 1:nb])
                ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
                rb[:, 1:nb] .= ub[:, 1:nb]
            end
        end

        # Expand subspace, new expanded Hc comes out
        Hc, nact = _jdb_expand!(V, W, Hc, rb, j, nb, kmax, A, orth_method, ub, nhl)
        nmv += nact
        j += nact

        # Diagonalize over full subspace (soft-locked pairs stay in V)
        @timing "jd: diag" begin
            ew, Vc = eigen(Hc)
        end
    end

    @printf("jd: done. nconv=%d/%d  iter=%d  nmv=%d  avg_hardlocked/k=%.2f\n", nconv, k, jd_iter, nmv, nhl_sum / (jd_iter * k))

    if nconv < k
        @warn "jd did not converge within $maxit iterations."
    end

    nhl = highest_hard_locked_ind
    lambda = vcat(lambda_hl[1:nhl], ew[1:k - nhl])
    X = similar(v0, n, k)
    X[:, 1:nhl] .= V[:, 1:nhl]
    X[:, nhl+1:k] .= V[:, nhl+1:j] * Vc[:, 1:k - nhl]

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Phase 1 (BLAS-3): orthogonalize all nb vectors at once against V[:,1:j].
Phase 2: orthonormalize the nb candidates among themselves via `orth_method`, accept non-degenerate columns.
Then compute A * new columns and returns the expanded projected Hamiltonian Hexp.

Returns the expanded projected Hamiltonian `Hexp` and the number of accepted vectors `nact`.
"""
@views function _jdb_expand!(V::AbstractMatrix, W::AbstractMatrix,
                      Hc::AbstractMatrix, rb::AbstractMatrix,
                      j::Int, nb::Int, kmax::Int, A,
                      orth_method::Symbol=:mgs,
                      buf::AbstractMatrix=similar(V, size(V, 1), nb),
                      nhl::Int=0)
    nact = 0

    @timing "jd: ortho" begin
        # Phase 1: orthogonalize all nb corrections against existing V[:,1:j]
        let Vj = view(V, :, 1:j), rbv = view(rb, :, 1:nb)
            c1v = Vj' * rbv
            mul!(rbv, Vj, c1v, -1, 1)
            if orth_method == :mgs2
                mul!(c1v, Vj', rbv)
                mul!(rbv, Vj, c1v, -1, 1)
            end
        end

        # Phase 2: orthonormalize nb candidates among themselves; dependent columns are dropped
        nact_rb = _ortho!(rb, nb, orth_method, buf)
        nact = min(nact_rb, kmax - j)
        if nact > 0
            V[:, j+1:j+nact] .= rb[:, 1:nact]
        end
    end

    nact == 0 && return Hc, 0

    # Compute A * new basis vectors
    @timing "jd: matvec" mul!(W[:, j+1:j+nact], A, V[:, j+1:j+nact])

    # Update projected Hamiltonian (full column block, BLAS-3)
    @timing "jd: expand overlap" begin
        jr = j - nhl
        Hexp = similar(Hc, jr+nact, jr+nact)
        Hexp[1:jr, 1:jr] .= Hc
        mul!(Hexp[:, jr+1:jr+nact], V[:, nhl+1:j+nact]', W[:, j+1:j+nact])
        Hexp = Hermitian(Hexp)  # Hermitian ==> no need for explicit symmetrization
    end

    return Hexp, nact
end
