using LinearAlgebra
using Printf


"""
    jd_sketched(A, v0; k=5, kwargs...)

Sketched Jacobi-Davidson eigensolver for symmetric/Hermitian matrices.
Structure mirrors `jd` exactly; the only differences are:
  - V is kept Θ-orthonormal (sketched orthogonalization) instead of standard orthonormal
  - The projected problem is the generalized Hermitian eigenproblem Mc y = λ Oc y, where
    Mc = V'AV and Oc = V'V (the Gram matrix of the Θ-orthonormal basis)
  - Eigenvalues are real and sorted by `eigen(Hermitian, Hermitian)`; no sortperm needed
  - X = V*U[:,1:k] is standard orthonormal since U'*Oc*U = I = U'*(V'V)*U = X'X
  - Convergence is measured in Θ-norm: ‖Θr‖ instead of ‖r‖; sketched residuals are
    computed as SW*U - SV*U*diag(θ) without re-applying Θ to the full-space residual

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
- `orth_method=:rcgs`: Θ-orthogonalization method (`:rcgs`, `:rcgs2`, `:rqr`)

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
                          sketch_type::String="sparsestack",
                          sketch_size::Int=-1,
                          orth_method::Symbol=:rcgs)

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


    T = eltype(v0)

    Theta = sketch(n, s, sketch_type, v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jd_sketched: allocation" begin
        # Active subspace: columns 1:j live in the the following buffers.
        V  = similar(v0, T, n, jmax) # search space
        W  = similar(v0, T, n, jmax) # W = A * V
        SV = fill!(similar(v0, T, s, jmax), zero(T)) # Sketch of search space
        SW = fill!(similar(v0, T, s, jmax), zero(T)) # Sketch of A * V

        # n_buffer/s_buffer each serve two non-overlapping roles: halves [1:kb] hold
        # Ritz vectors / residuals / corrections; full [1:2kb] used as rotation scratch on restart.
        n_buffer = similar(v0, T, n, 2*kb)
        ub    = view(n_buffer, :, 1:kb)        # Ritz vectors X_b = V * U[:,active]
        rb    = view(n_buffer, :, kb+1:2*kb)   # residuals r_b; repurposed as corrections after precond
        s_buffer = similar(v0, T, s, 2*kb)
        SX_rq = view(s_buffer, :, 1:kb)        # Θ * X_b (for sketched residual norms)
        SW_rq = view(s_buffer, :, kb+1:2*kb)   # Θ * A * X_b; reused as sketch scratch during expand
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Mc: projected matrix V'AV, j×j Hermitian
    # Oc: Gram matrix V'V,       j×j Hermitian (≈ I but not exactly for Θ-orthonormal V)
    # ew: Ritz values from generalized Hermitian eigen, length j, Real, sorted
    # U:  Ritz vectors (Oc-orthonormal), j×j

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
        # Θ-orthonormalize in-place; pre-compute sketch first, SW_rq used as scratch.
        mul!(SV[:, 1:j], Theta, V[:, 1:j])
        j = _sketch_ortho!(V[:, 1:j], SV[:, 1:j], orth_method)
    end
    @timing "jd_sketched: matvec" begin
        mul!(W[:, 1:j], A, V[:, 1:j])
        nmv += j
    end
    @timing "jd_sketched: sketch" begin
        mul!(SW[:, 1:j], Theta, W[:, 1:j])
    end
    # Mc = V'AV, Oc = V'V — stored as plain arrays so similar(Mc,...) preserves device type;
    # Hermitian wrapper applied only at eigen call sites.
    @timing "jd_sketched: overlap" begin
        Mc = Hermitian(V[:, 1:j]' * W[:, 1:j])
        Oc = Hermitian(V[:, 1:j]' * V[:, 1:j])
    end

    # Generalized Hermitian eigenproblem Mc y = λ Oc y.
    # Eigenvalues are real and sorted; eigenvectors are Oc-orthonormal (U' Oc U = I).
    @timing "jd_sketched: diag" begin
        F  = eigen(Hermitian(Mc), Hermitian(Oc))
        ew = F.values
        U  = F.vectors
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: keep jmin Ritz vectors (soft-locked pairs stay in V).
        # U[:,1:nk] is Oc-orthonormal, so after rotation V_new = V*U[:,1:nk]:
        #   Mc_new = Diagonal(ew[1:nk])  and  Oc_new = I.
        # No QR needed (unlike the general-eigen restart in the old code).
        if j + nb >= jmax
            @timing "jd_sketched: restart" begin
                nk    = min(jmin, j)
                mul!(n_buffer[:, 1:nk], V[:, 1:j],  U[:, 1:nk])
                V[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SV[:, 1:j], U[:, 1:nk])
                SV[:, 1:nk] .= s_buffer[:, 1:nk]
                mul!(n_buffer[:, 1:nk], W[:, 1:j],  U[:, 1:nk])
                W[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SW[:, 1:j], U[:, 1:nk])
                SW[:, 1:nk] .= s_buffer[:, 1:nk]
                j  = nk
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                Mc = similar(V, T, nk, nk)
                fill!(Mc, zero(T))
                Mc[diagind(Mc)] .= ew[1:nk]
                Mc = Hermitian(Mc)
                Oc = similar(V, T, nk, nk)
                fill!(Oc, zero(T))
                Oc[diagind(Oc)] .= one(T)
                Oc = Hermitian(Oc)
                U  = similar(V, T, nk, nk)
                fill!(U,  zero(T)); U[diagind(U)] .= one(T)
                ew = ew[1:nk]
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (BLAS-3).
        # ew[nconv+1:nconv+nb] are used directly as Rayleigh quotients.
        @timing "jd_sketched: residual" begin
            Y_nb = view(U, :, nconv+1:nconv+nb)
            mul!(ub[:, 1:nb], V[:, 1:j], Y_nb)
            mul!(rb[:, 1:nb], W[:, 1:j], Y_nb)
            theta_b = ew[nconv+1:nconv+nb]
            rb[:, 1:nb] .-= ub[:, 1:nb] .* theta_b'
        end

        # Compute sketched residuals in sketch space: Θr = SW*U - SV*U*diag(θ).
        # Avoids re-applying Θ to the full-space residual rb.
        @timing "jd_sketched: sketch" begin
            mul!(SX_rq[:, 1:nb], SV[:, 1:j], Y_nb)
            mul!(SW_rq[:, 1:nb], SW[:, 1:j], Y_nb)
            SW_rq[:, 1:nb] .-= SX_rq[:, 1:nb] .* theta_b'
            rnorms = columnwise_norms(SW_rq[:, 1:nb])
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
                first_notconv = findfirst(rnorms[1:nb_target] .>= tol)
                nconv_new = isnothing(first_notconv) ? nb_target : first_notconv - 1
            end
        end

        # Lock newly converged pairs (Ritz vectors stay in V via soft-locking)
        if nconv_new > 0
            @timing "jd_sketched: lock" begin
                nconv += nconv_new
                nconv >= k && break
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

        # Expand subspace; SW_rq used as sketch scratch for corrections
        Mc, Oc, nact = _jdrb_expand!(V, SV, W, SW, Mc, Oc, rb, j, nb, jmax, A, Theta,
                                      orth_method, SW_rq)
        nmv += nact
        j   += nact

        # Generalized Hermitian eigenproblem Mc y = λ Oc y; eigenvalues real and sorted.
        @timing "jd_sketched: diag" begin
            F  = eigen(Hermitian(Mc), Hermitian(Oc))
            ew = F.values
            U  = F.vectors
        end
    end

    disp && @printf("jd_sketched: done. nconv=%d/%d  iter=%d  nmv=%d\n", nconv, k, jd_iter, nmv)

    if nconv < k
        @warn "jd_sketched did not converge within $maxit iterations."
    end

    # U[:,1:k] are Oc-orthonormal => X = V*U[:,1:k] is standard orthonormal
    return V[:, 1:j] * U[:, 1:k], copy(ew[1:k]), history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Mirrors `_jdb_expand!` from jd, but uses Θ-orthogonalization and maintains
Mc = V'AV, Oc = V'V, and SW = Θ*AV incrementally.

Phase 1: sketch rb, Θ-project out V[:,1:j], then Θ-orthonormalize among themselves.
Then compute A * new columns, update SW, and expand Mc and Oc.

`SV_scratch` (s×kb) is a workspace for sketching corrections; reuse SW_rq (not live during expand).

Returns the expanded Mc, Oc, and the number of accepted vectors `nact`.
"""
@views function _jdrb_expand!(V, SV, W, SW, Mc, Oc, rb, j, nb, jmax, A, Theta, orth_method,
                               SV_scratch)
    @timing "jd_sketched: ortho" begin
        mul!(SV_scratch[:, 1:nb], Theta, rb[:, 1:nb])
        _sketch_project_out!(rb[:, 1:nb], SV_scratch[:, 1:nb], V[:, 1:j], SV[:, 1:j], orth_method)
        nact = _sketch_ortho!(rb[:, 1:nb], SV_scratch[:, 1:nb], orth_method)
    end

    nact == 0 && return Mc, Oc, 0
    nact = min(nact, jmax - j)
    nact == 0 && return Mc, Oc, 0

    # Commit accepted columns into V/SV
    V[:,  j+1:j+nact] .= rb[:,         1:nact]
    SV[:, j+1:j+nact] .= SV_scratch[:, 1:nact]

    # Compute A * new basis vectors and sketch AV
    @timing "jd_sketched: matvec" mul!(W[:, j+1:j+nact], A, V[:, j+1:j+nact])
    @timing "jd_sketched: sketch" mul!(SW[:, j+1:j+nact], Theta, W[:, j+1:j+nact])

    # Expand Mc = V'AV and Oc = V'V
    @timing "jd_sketched: expand overlap" begin
        Mexp = similar(Mc, j+nact, j+nact)
        Mexp[1:j, 1:j] .= Mc
        mul!(Mexp[:, j+1:j+nact], V[:, 1:j+nact]', W[:, j+1:j+nact])
        Mexp = Hermitian(Mexp)

        Oexp = similar(Oc, j+nact, j+nact)
        Oexp[1:j, 1:j] .= Oc
        mul!(Oexp[:, j+1:j+nact], V[:, 1:j+nact]', V[:, j+1:j+nact])
        Oexp = Hermitian(Oexp)
    end

    return Mexp, Oexp, nact
end
