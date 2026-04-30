using LinearAlgebra
using Printf


"""
    jd_sketched(A, v0; k=5, kwargs...)

Sketched Jacobi-Davidson eigensolver for symmetric/Hermitian matrices.
Structure mirrors `jd` exactly; the only differences are:
  - V is kept Θ-orthonormal (sketched orthogonalization) instead of standard orthonormal
  - Sketch-orthogonality ensures cond(V'V) ≤ (1+ε)/(1-ε), bounding diagonalization error
  - The projected problem is the generalized Hermitian eigenproblem: Hc x = θ Sc x,
    where Hc = V'AV (Hermitian) and Sc = V'V (Hermitian PD, well-conditioned by sketch)
  - Convergence is measured in standard L2 norm: ‖r‖

Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. nbuff=2 buffer pairs are kept beyond k. Search space restarts
to jmin=2*(k+nbuff) vectors when full, and grows up to jmax=4*(k+nbuff).

# Arguments
- `A`: Symmetric/Hermitian matrix or operator
- `v0`: Initial vectors (n x m matrix)
- `k`: Number of eigenpairs (default 5)

# Keyword Arguments
- `tol=1e-8`: Residual L2 norm convergence threshold
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


    T = complex(eltype(v0))

    Theta = sketch(n, s, sketch_type, v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jd_sketched: allocation" begin
        # Active subspace: columns 1:j live in the the following buffers.
        V  = similar(v0, T, n, jmax)
        W  = similar(v0, T, n, jmax)
        SV = similar(v0, T, s, jmax); SV .= zero(T)  # arrays potentially holding result of a sparse matrix
                                                      # multiplication must be initialized to zero for safety
        # Work buffers for various allocation free operations:
        n_buffer = similar(v0, T, n, jmax)
        s_buffer = similar(v0, T, s, jmax)
        # Ritz vectors and residuals for active pairs (like ub/rb in jdsym_block)
        # ub is also reused as T_corr_buf during expand (never live at the same time)
        ub      = similar(v0, T, n, kb)
        rb      = similar(v0, T, n, kb)
        # Scratch buffers reused as ST_corr_buf/SV_scratch in _jdrb_expand!
        # (never live at the same time as the expand call)
        s_scratch1 = similar(v0, T, s, kb); s_scratch1 .= zero(T)
        s_scratch2 = similar(v0, T, s, kb); s_scratch2 .= zero(T)
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Hc: projected Hamiltonian, j x j, Hc = V'AV (Hermitian)
    # Sc: projected overlap,     j x j, Sc = V'V  (Hermitian PD, well-conditioned by Θ-ortho)
    # U:  eigenvectors of the projected problem, j x j, U'Sc U = I
    # ew: eigenvalues of the projected problem, length j, Real vector

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
        # s_scratch2 (s×kb) used as SV_buf scratch; not yet populated.
        j = _sketch_ortho!(V[:, 1:j], SV[:, 1:j], V[:, 1:j],
                             Theta, orth_method,
                             s_scratch2)
    end
    @timing "jd_sketched: matvec" begin
        mul!(W[:, 1:j], A, V[:, 1:j])
        nmv += j
    end
    # Hc = V'AV (Hermitian), Sc = V'V (Hermitian PD, well-conditioned by Θ-ortho)
    @timing "jd_sketched: overlap" begin
        Hc = V[:, 1:j]' * W[:, 1:j]
        Sc = V[:, 1:j]' * V[:, 1:j]
    end

    # Generalized Hermitian eigenproblem: Hc x = θ Sc x
    # Eigenvalues are real and exact Rayleigh quotients; U satisfies U'Sc U = I,
    # so Ritz vectors V*U are standard-orthonormal.
    @timing "jd_sketched: diag" begin
        F    = eigen(Hermitian(Hc), Hermitian(Sc))
        ew   = F.values
        U    = F.vectors
        perm = sortperm(ew)
        ew   = copy(ew[perm])
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
                # U columns satisfy U'Sc U = I, so V*U[:,1:nk] is standard-orthonormal.
                # Use them directly as the restarted basis (no QR needed).
                Q_rst = U[:, 1:nk]
                mul!(n_buffer[:, 1:nk], V[:, 1:j],  Q_rst)
                V[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SV[:, 1:j], Q_rst)
                SV[:, 1:nk] .= s_buffer[:, 1:nk]
                mul!(n_buffer[:, 1:nk], W[:, 1:j],  Q_rst)
                W[:, 1:nk] .= n_buffer[:, 1:nk]
                j  = nk
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                # Rotate Hc and Sc to the new basis.
                # After restart V*U is orthonormal, so Sc_new ≈ I and Hc_new ≈ diag(ew).
                Hc = Q_rst' * Hc * Q_rst   # nk×nk
                Sc = Q_rst' * Sc * Q_rst   # nk×nk
                F_rst = eigen(Hermitian(Hc), Hermitian(Sc))
                ew    = F_rst.values
                U     = F_rst.vectors
                perm  = sortperm(ew)
                ew    = copy(ew[perm]);  U = copy(U[:, perm])
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Compute residuals for active pairs nconv+1..nconv+nb (BLAS-3, like jd)
        @timing "jd_sketched: residual" begin
            Y_nb    = view(U, :, nconv+1:nconv+nb)
            mul!(ub[:, 1:nb], V[:, 1:j], Y_nb)
            mul!(rb[:, 1:nb], W[:, 1:j], Y_nb)
            # ew are exact Rayleigh quotients from the generalized Hermitian eigenproblem
            theta_b = ew[nconv+1:nconv+nb]
            # r = AX - X*diag(theta), in-place
            @views rb[:, 1:nb] .-= ub[:, 1:nb] .* theta_b'
            rnorms = Array(columnwise_norms(rb[:, 1:nb]))
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

        # Expand subspace: Θ-orthogonalize corrections against V, return expanded Hc and Sc.
        # ub=T_corr_buf, s_scratch1=ST_corr_buf, s_scratch2=SV_scratch — merged buffers, not live here
        Hc, Sc, nact = _jdrb_expand!(V, SV, W, Hc, Sc, rb, j, nb, jmax, A, Theta, orth_method,
                                      ub, s_scratch1, s_scratch2)
        nmv += nact
        j   += nact

        # Generalized Hermitian diagonalization of expanded Hc, Sc
        @timing "jd_sketched: diag" begin
            F    = eigen(Hermitian(Hc), Hermitian(Sc))
            ew   = F.values
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

    # U satisfies U'Sc U = I, so V*U[:,1:k] is exactly standard-orthonormal.
    # No Cholesky cleanup or final re-diagonalization needed.
    @timing "jd_sketched: finalize" begin
        U_fin = U[:, 1:k]                                    # j×k
        mul!(n_buffer[:, 1:k], V[:, 1:j], U_fin)
        lambda = ew[1:k]
        # Final output: only this n×k allocation is unavoidable.
        # Convert back to input type: real inputs give real outputs (imaginary parts are ~0).
        X_full = n_buffer[:, 1:k]
        perm_s = sortperm(lambda)
        X_sorted = copy(X_full[:, perm_s])
        lambda   = copy(lambda[perm_s])
        if eltype(v0) <: Real
            # Eigenvectors computed in complex arithmetic carry an arbitrary scalar phase
            # e^(iφ) per column. Remove it so that real.() discards only imaginary noise.
            absvals = abs.(X_sorted)
            maxabs, indices = findmax(absvals; dims=1)
            X_sorted ./= X_sorted[indices] ./ maxabs
            X = real.(X_sorted)
        else
            X = X_sorted
        end
    end

    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Mirrors `_jdb_expand!` from jd, but uses Θ-orthogonalization instead of MGS.

Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j].
Phase 2: Θ-orthonormalize among the nb new vectors; normalize; accept.
Then compute A * new columns and expand Hc = V'AV and Sc = V'V incrementally (BLAS-3).

Returns the expanded Hc, Sc and the number of accepted vectors `nact`.
"""
@views function _jdrb_expand!(V, SV, W, Hc, Sc, rb, j, nb, jmax, A, Theta, orth_method,
                       T_corr_buf, ST_corr_buf, SV_scratch)
    # T_corr_buf  = ub         (n×kb, reused from Ritz-vector buffer — not live during expand)
    # ST_corr_buf = s_scratch1 (s×kb, reused from scratch buffer     — not live during expand)
    # SV_scratch  = s_scratch2 (s×kb, reused from scratch buffer     — not live during expand)

    # Phase 1: Θ-orthogonalize all nb corrections against V[:,1:j] — in-place, no alloc
    @timing "jd_sketched: ortho" begin
        rbb = view(rb, :, 1:nb)
        _sketch_deflate!(rbb, V[:, 1:j], SV[:, 1:j], Theta, orth_method, SV_scratch)
    end

    # Phase 2: Θ-orthonormalize among the nb corrections into pre-allocated buffers
    @timing "jd_sketched: ortho" begin
        nact = _sketch_ortho!(T_corr_buf, ST_corr_buf, rb[:, 1:nb], Theta,
                                orth_method, SV_scratch)
    end

    nact == 0 && return Hc, Sc, 0
    nact = min(nact, jmax - j)
    nact == 0 && return Hc, Sc, 0

    T_corr  = view(T_corr_buf,  :, 1:nact)
    ST_corr = view(ST_corr_buf, :, 1:nact)

    # Compute A * new basis vectors; store sketch of new V columns
    @timing "jd_sketched: matvec" mul!(W[:, j+1:j+nact], A, T_corr)
    @timing "jd_sketched: sketch" SV[:, j+1:j+nact] .= ST_corr

    # Expand Hc = V'AV and Sc = V'V incrementally (BLAS-3).
    # Exploit Hermitian structure: lower-left block = conjugate transpose of upper-right.
    @timing "jd_sketched: overlap" begin
        W_new = view(W, :, j+1:j+nact)
        Hexp = similar(Hc, j+nact, j+nact)
        Hexp[1:j, 1:j] .= Hc
        mul!(Hexp[1:j,        j+1:j+nact], V[:, 1:j]', W_new)
        Hexp[j+1:j+nact, 1:j] .= Hexp[1:j, j+1:j+nact]'   # A Hermitian
        mul!(Hexp[j+1:j+nact, j+1:j+nact], T_corr', W_new)

        Scexp = similar(Sc, j+nact, j+nact)
        Scexp[1:j, 1:j] .= Sc
        mul!(Scexp[1:j,        j+1:j+nact], V[:, 1:j]', T_corr)
        Scexp[j+1:j+nact, 1:j] .= Scexp[1:j, j+1:j+nact]'
        mul!(Scexp[j+1:j+nact, j+1:j+nact], T_corr', T_corr)
    end

    # Write T_corr into V buffer
    @timing "jd_sketched: expand" V[:, j+1:j+nact] .= T_corr

    return Hexp, Scexp, nact
end