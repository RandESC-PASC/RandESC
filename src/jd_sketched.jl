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
  - Convergence is measured as norm of residuals.

Soft-locking: converged Ritz vectors stay in V; corrections are generated only
for active pairs. Hard-locking: at each restart, soft-locked pairs whose eigenvalue
is more than `hardlock_gap` below the largest soft-locked eigenvalue are moved into
a fixed prefix V[:,1:nhl] and excluded from the projected eigenvalue problem,
reducing its size. The restart already rotates V to the Ritz basis (standard
orthonormal), so no extra rotation is needed for hard-locking. nbuff=2 buffer
pairs are kept beyond k. Search space restarts to jmin=2*(k+nbuff) vectors when
full, and grows up to jmax=4*(k+nbuff).

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
- `hardlock_gap=0.01`: Minimum gap (largest_soft_λ - λᵢ) required to hard-lock pair i

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
                          orth_method::Symbol=:rcgs,
                          hardlock_gap::Float64=0.01,
                          sketch_project_out::Bool=false)

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
        V  = similar(v0, T, n, jmax) # search space (hard-locked prefix + active)
        W  = similar(v0, T, n, jmax) # W = A * V
        SV = fill!(similar(v0, T, s, jmax), zero(T)) # Sketch of search space

        lambda_hl = fill!(similar(v0, real(T), k), zero(real(T)))  # eigenvalues of hard-locked pairs

        # n_buffer/s_buffer: rotation scratch during restart (full 2*kb columns);
        # ub/rb are declared as views into these buffers at each usage site.
        n_buffer = similar(v0, T, n, 2*kb)
        s_buffer = similar(v0, T, s, 2*kb)
    end

    # Other arrays used in the solver, allocated on the fly (small compared to the above):
    # Mc: projected matrix V'AV over reduced space, jr×jr Hermitian
    # Oc: Gram matrix V'V over reduced space,       jr×jr Hermitian (≈ I but not exactly)
    # ew: Ritz values from generalized Hermitian eigen, length jr, Real, sorted
    # U:  Ritz vectors (Oc-orthonormal), jr×jr

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0
    highest_hard_locked_ind = 0  # nhl: V[:,1:nhl] are hard-locked eigenvectors

    # ── Initialize subspace ───────────────────────────────────────────────
    j = min(kb, n)
    @timing "jd_sketched: init" begin
        nc = min(size(v0, 2), j)
        V[:, 1:nc] .= v0[:, 1:nc]
        if nc < j
            randn!(TaskLocalRNG(), V[:, nc+1:j])   # in-place, no temp allocation
        end
        # Θ-orthonormalize in-place; pre-compute sketch first.
        mul!(SV[:, 1:j], Theta, V[:, 1:j])
        j = _sketch_ortho!(V[:, 1:j], SV[:, 1:j], orth_method)
    end
    @timing "jd_sketched: matvec" begin
        mul!(W[:, 1:j], A, V[:, 1:j])
        nmv += j
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

        nhl = highest_hard_locked_ind
        nconv_r = nconv - nhl  # soft-locked count in reduced space V[:,nhl+1:j]

        # Active pairs: nconv+1 .. nconv+nb  (k-nconv target + nbuff buffer)
        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: rotate reduced space to Ritz basis, then opportunistically hard-lock.
        # Hard-locking is free here since V is already standard orthonormal after rotation
        # (U' Oc U = I  =>  (VU)'(VU) = I), so SV[:,nhl+1:...] = Theta*V[:,nhl+1:...] exactly.
        if j + nb >= jmax
            @timing "jd_sketched: restart" begin
                nk = min(jmin, j - nhl)
                mul!(n_buffer[:, 1:nk], V[:, nhl+1:j],  U[:, 1:nk])
                V[:, nhl+1:nhl+nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SV[:, nhl+1:j], U[:, 1:nk])
                SV[:, nhl+1:nhl+nk] .= s_buffer[:, 1:nk]
                mul!(n_buffer[:, 1:nk], W[:, nhl+1:j],  U[:, 1:nk])
                W[:, nhl+1:nhl+nk] .= n_buffer[:, 1:nk]
                ew = ew[1:nk]

                # Hard lock: V[:,nhl+1:nhl+nconv_r] are now Ritz vectors; pairs
                # with gap > hardlock_gap to the largest soft-locked one move into prefix.
                if nconv_r >= 2
                    nhardlocknew = findfirst(ew[nconv_r:nconv_r] .- ew[1:nconv_r] .< hardlock_gap)
                    nhardlocknew = isnothing(nhardlocknew) ? nconv_r : nhardlocknew - 1
                else
                    nhardlocknew = 0
                end
                if nhardlocknew > 0
                    lambda_hl[nhl+1:nhl+nhardlocknew] .= ew[1:nhardlocknew]
                    highest_hard_locked_ind += nhardlocknew
                    nhl = highest_hard_locked_ind
                    ew = ew[nhardlocknew+1:end]
                    nconv_r = nconv - nhl
                end

                jr = length(ew)
                j  = nhl + jr
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                Mc = similar(V, T, jr, jr)
                fill!(Mc, zero(T))
                Mc[diagind(Mc)] .= ew
                Mc = Hermitian(Mc)
                Oc = similar(V, T, jr, jr)
                fill!(Oc, zero(T))
                Oc[diagind(Oc)] .= one(T)
                Oc = Hermitian(Oc)
                U  = similar(V, T, jr, jr)
                fill!(U,  zero(T)); U[diagind(U)] .= one(T)
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d  nhl=%d\n", j, nconv, k, nhl)
                restarted = true
            end
        else
            restarted = false
        end

        # Compute residuals for active pairs (BLAS-3).
        # Indices in reduced space: nconv_r+1 .. nconv_r+nb
        ub    = view(n_buffer, :, 1:nb)
        rb    = view(n_buffer, :, nb+1:2*nb)
        @timing "jd_sketched: residual" begin
            Y_nb = view(U, :, nconv_r+1:nconv_r+nb)
            if restarted
                # U is identity, can simply copy columns
                ub[:, 1:nb] .= V[:, nhl+nconv_r+1:nhl+nconv_r+nb]
                rb[:, 1:nb] .= W[:, nhl+nconv_r+1:nhl+nconv_r+nb]
            else
                mul!(ub[:, 1:nb], V[:, nhl+1:j], Y_nb)
                mul!(rb[:, 1:nb], W[:, nhl+1:j], Y_nb)
            end
            rb[:, 1:nb] .-= ub[:, 1:nb] .* ew[nconv_r+1:nconv_r+nb]'
            rnorms = columnwise_norms(rb[:, 1:nb])
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

        scratch = view(s_buffer, :, nb+1:2*nb)
        Mc, Oc, nact = _jdrb_expand!(V, SV, W, Mc, Oc, rb, j, nb, jmax, A, Theta,
                                      orth_method, scratch, nhl, sketch_project_out)
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

    nhl = highest_hard_locked_ind
    lambda = vcat(lambda_hl[1:nhl], ew[1:k - nhl])
    X = similar(v0, n, k)
    X[:, 1:nhl] .= V[:, 1:nhl]
    X[:, nhl+1:k] .= V[:, nhl+1:j] * U[:, 1:k - nhl]
    return X, lambda, history[1:hist_row, :]
end


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Expand search subspace with up to `nb` correction vectors from `rb[:,1:nb]`.

Mirrors `_jdb_expand!` from jd, but uses Θ-orthogonalization and maintains
Mc = V'AV, Oc = V'V incrementally over the reduced space V[:,nhl+1:j+nact].
The full V[:,1:j] (including hard-locked prefix) is used for orthogonalization
so new vectors are orthogonal to everything.

Phase 1: sketch rb, Θ-project out V[:,1:j], then Θ-orthonormalize among themselves.
Then compute A * new columns and expand Mc and Oc.

`SV_scratch` (s×kb) is a workspace for sketching corrections

Returns the expanded Mc, Oc, and the number of accepted vectors `nact`.
"""
@views function _jdrb_expand!(V, SV, W, Mc, Oc, rb, j, nb, jmax, A, Theta, orth_method,
                               SV_scratch, nhl=0, sketch_project_out=false)
    @timing "jd_sketched: ortho" begin
        if sketch_project_out
            mul!(SV_scratch[:, 1:nb], Theta, rb[:, 1:nb])
            _sketch_project_out!(rb[:, 1:nb], SV_scratch[:, 1:nb], V[:, 1:j], SV[:, 1:j], orth_method)
        else
            c = V[:, 1:j]' * rb[:, 1:nb]
            mul!(rb[:, 1:nb], V[:, 1:j], c, -1, 1)
            mul!(SV_scratch[:, 1:nb], Theta, rb[:, 1:nb])
        end
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

    # Expand Mc = V'AV and Oc = V'V over reduced space V[:,nhl+1:j+nact]
    @timing "jd_sketched: expand overlap" begin
        jr = j - nhl
        Mexp = similar(Mc, jr+nact, jr+nact)
        Mexp[1:jr, 1:jr] .= Mc
        mul!(Mexp[:, jr+1:jr+nact], V[:, nhl+1:j+nact]', W[:, j+1:j+nact])
        Mexp = Hermitian(Mexp)

        Oexp = similar(Oc, jr+nact, jr+nact)
        Oexp[1:jr, 1:jr] .= Oc
        mul!(Oexp[:, jr+1:jr+nact], V[:, nhl+1:j+nact]', V[:, j+1:j+nact])
        Oexp = Hermitian(Oexp)
    end

    return Mexp, Oexp, nact
end
