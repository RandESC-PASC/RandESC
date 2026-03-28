using LinearAlgebra
using Printf

"""
    jdsym_block(A; k=5, kwargs...)

Davidson eigensolver for symmetric/Hermitian matrices (smallest eigenvalues).
All kb = k + nbuff Ritz vectors always participate in the Ritz diagonalization.
Correction vectors are generated for unconverged pairs among the first k, plus all
nbuff buffer pairs. Convergence is checked via residual norm ||r_i|| < tol per pair.

Algorithm outline per iteration:
  1. Diagonalize projected matrix Hc → (ew, Vc)
  2. Compute residuals r_i = A*u_i - ew_i*u_i for all kb pairs
  3. Check convergence: ||r_i|| < tol for i=1..k independently
  4. Restart subspace if j + notcnv_kb > kmax  (rebuild from kb current Ritz vectors)
  5. Pack unconverged corrections; apply preconditioner
  6. Expand subspace: MGS2 against V, normalize, update Hc = V'*AV

# Arguments
- `A`: Symmetric/Hermitian matrix (dense or sparse)
- `k`: Number of eigenpairs to compute (default: 5)

# Keyword Arguments
- `tol=1e-8`: Convergence tolerance; pair i converges when ||r_i|| < tol
- `maxit=100`: Maximum outer iterations
- `nblock=-1`: Unused (kept for API compatibility)
- `nbuff=0`: Number of buffer vectors beyond k; block size is kb = k + nbuff
- `kmax=-1`: Max subspace dimension (default: `2*kb + 10`)
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
                     nbuff::Int=15,
                     kmax::Int=-1,
                     v0=nothing,
                     M=nothing,
                     precond_preparator=nothing,
                     disp::Bool=false,
                     debug::Bool=false)

    n = size(A, 1)
    k = min(k, n)
    kb = min(k + nbuff, n)   # block size: target + buffer

    if n < 1
        return zeros(Float64, 0, 0), Float64[], zeros(Float64, 0, 3)
    end

    kmax = kmax < 0 ? min(n, 2kb + 10) : kmax

    if kb > kmax ÷ 2
        error("kmax too small: need kmax > 2*(k+nbuff) (kb=$kb, kmax=$kmax)")
    end

    T = isnothing(v0) ? Float64 : eltype(v0)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jdsym_block: allocation" begin
        V    = zeros(T, n, kmax)    # search space basis
        W    = zeros(T, n, kmax)    # A * V
        Hc   = zeros(T, kmax, kmax) # projected Hamiltonian: V' * W
        Vc   = zeros(T, kmax, kmax) # eigenvectors of projected problem
        ew   = zeros(Float64, kmax) # eigenvalues of projected problem
        ub   = zeros(T, n, kb)      # Ritz vectors for correction
        rb   = zeros(T, n, kb)      # residuals / corrections
        Vtmp = zeros(T, n, kb)      # temporary for restart
        Wtmp = zeros(T, n, kb)      # temporary for restart
        lambda  = zeros(Float64, k)
        rnorms  = zeros(Float64, kb)
        conv    = falses(k)
        c_mgs   = zeros(T, kmax)
        c_exp1  = zeros(T, kmax, kb)
        c_exp2  = zeros(T, kb)
    end

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
    notcnv = k
    for iter in 1:maxit
        jd_iter = iter

        # Diagonalize projected EVP
        @timing "jdsym_block: diag" begin
            @views F = eigen(Hermitian(Hc[1:j, 1:j]))
            ew[1:j]      .= F.values
            Vc[1:j, 1:j] .= F.vectors
        end

        # Compute all kb Ritz vectors and residuals (blocked BLAS-3)
        @timing "jdsym_block: residual" begin
            @views mul!(ub[:, 1:kb], V[:, 1:j], Vc[1:j, 1:kb])
            @views mul!(rb[:, 1:kb], W[:, 1:j], Vc[1:j, 1:kb])
            @views rb[:, 1:kb] .-= ub[:, 1:kb] .* ew[1:kb]'
            for i in 1:kb; rnorms[i] = norm(view(rb, :, i)); end
        end

        # Check convergence for first k pairs
        notcnv = 0
        for i in 1:k
            conv[i] = rnorms[i] < tol
            conv[i] || (notcnv += 1)
        end

        notcnv == 0 && break

        # Number of corrections: unconverged among 1:k + all buffer pairs
        notcnv_kb = notcnv + (kb - k)

        # Restart: rebuild from kb current Ritz vectors; recompute residuals after
        if j + notcnv_kb > kmax
            @timing "jdsym_block: restart" begin
                @views mul!(Vtmp[:, 1:kb], V[:, 1:j], Vc[1:j, 1:kb])
                @views mul!(Wtmp[:, 1:kb], W[:, 1:j], Vc[1:j, 1:kb])
                @views V[:, 1:kb] .= Vtmp[:, 1:kb]
                @views W[:, 1:kb] .= Wtmp[:, 1:kb]
                Hc[1:kmax, 1:kmax] .= zero(T)
                for i in 1:kb; Hc[i, i] = ew[i]; end
                j = kb
                @views F = eigen(Hermitian(Hc[1:j, 1:j]))
                ew[1:j]      .= F.values
                Vc[1:j, 1:j] .= F.vectors
                (disp || debug) && @printf("  RESTART -> j=%d  notcnv=%d/%d\n", j, notcnv, k)
            end
            @timing "jdsym_block: residual" begin
                @views mul!(ub[:, 1:kb], V[:, 1:j], Vc[1:j, 1:kb])
                @views mul!(rb[:, 1:kb], W[:, 1:j], Vc[1:j, 1:kb])
                @views rb[:, 1:kb] .-= ub[:, 1:kb] .* ew[1:kb]'
            end
        end

        # Pack unconverged corrections into contiguous slots for expand
        np = 0
        for i in 1:kb
            i <= k && conv[i] && continue
            np += 1
            if np != i
                @views rb[:, np] .= rb[:, i]
                @views ub[:, np] .= ub[:, i]
            end
        end

        hist_row += 1
        history[hist_row, :] .= (maximum(view(rnorms, 1:k)), iter, nmv)

        if disp
            @printf("jdsym_block it=%4d  notcnv=%d/%d  j=%d  |r|=%.2e..%.2e\n",
                    iter, notcnv, k, j,
                    minimum(view(rnorms, 1:k)), maximum(view(rnorms, 1:k)))
        end
        if debug
            ew_str = k <= 6 ? join([@sprintf("%.4e", ew[i]) for i in 1:k], " ") :
                               @sprintf("%.4e..%.4e", ew[1], ew[k])
            @printf("DEBUG it=%4d  notcnv=%d/%d  kb=%d  j=%d  |r|=%.2e..%.2e  ew=[%s]\n",
                    iter, notcnv, k, kb, j,
                    minimum(view(rnorms, 1:k)), maximum(view(rnorms, 1:k)), ew_str)
        end

        # Apply preconditioner
        @timing "jdsym_block: correction" begin
            if !isnothing(M)
                !isnothing(precond_preparator) && precond_preparator(M, view(ub, :, 1:notcnv_kb))
                for ip in 1:notcnv_kb
                    @views rb[:, ip] .= M \ rb[:, ip]
                end
            end
        end

        # Expand subspace
        nact = _jdb_expand!(V, W, Hc, rb, j, notcnv_kb, kmax, A, c_exp1, c_exp2)
        nmv += nact
        j += nact
        debug && nact < notcnv_kb && @printf("DEBUG           expand: only %d/%d vectors accepted (j=%d)\n", nact, notcnv_kb, j)

    end

    disp && @printf("jdsym_block: done. notcnv=%d  iter=%d  nmv=%d\n", notcnv, jd_iter, nmv)

    if !all(conv)
        @warn "jdsym_block did not converge within $maxit iterations."
    end

    # Compute final Ritz vectors from current basis
    @views Fout = eigen(Hermitian(Hc[1:j, 1:j]))
    @views mul!(Vtmp[:, 1:k], V[:, 1:j], Fout.vectors[1:j, 1:k])
    @views V[:, 1:k] .= Vtmp[:, 1:k]
    lambda .= Fout.values[1:k]

    return view(V, :, 1:k), lambda, history[1:hist_row, :]
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
