using LinearAlgebra
using Printf

"""
    jd_sketched_standard(A, v0; k=5, kwargs...)

Sketched Jacobi-Davidson eigensolver using the **standard (non-generalized)** projected
eigenproblem.  The search subspace is kept Θ-orthonormal and the projected matrix is

    Mc = (ΘV)ᴴ(ΘAV) = SV' SW

which is approximately Hermitian (since A is Hermitian).  The inner eigenproblem is
solved as a general eigen(Mc), eigenvalues sorted by real part.

The subspace is evolved in complex arithmetic even for real inputs (`T = complex(eltype(v0))`),
which keeps Mc approximately Hermitian and avoids sorting issues with nearly-degenerate
eigenvalues.  Real inputs get real output via a phase-normalization step in the finalize.

Compare with `jd_sketched`, which solves the generalized Hermitian problem `eigen(V'AV, V'V)`.
This variant avoids forming the Gram matrix Oc = V'V but requires an extra SW = ΘAV buffer
and a Cholesky re-orthonormalization finalize step.

Only sketch types that produce complex sketch matrices are compatible with real inputs here:
`sparsestack`, `sparsesign`, `complex_gaussian`, `complex_srtt`.  Using `real_gaussian` or
`real_srtt` with a real input will throw an `ArgumentError`.

Unlike `jd_sketched`, the projected matrix Mc = SV'SW is computed entirely from the sketch,
so sketch precision directly affects convergence.  The default is `sketch_prec=Float64`;
using `Float32` will typically stall convergence around 1e-4.

# Arguments
- `A`: Symmetric/Hermitian matrix or operator
- `v0`: Initial vectors (n × m matrix)
- `k`: Number of eigenpairs (default 5)

# Keyword Arguments
- `tol=1e-8`: Residual Θ-norm convergence threshold
- `maxit=200`: Maximum iterations
- `M=nothing`: Preconditioner, applied as `M \\ r`
- `precond_preparator=nothing`: Callback `f(M, X)` to refresh preconditioner
- `disp=false`: Print iteration info
- `sketch_type="sparsestack"`: Sketch operator type; for real inputs only
  `sparsestack`, `sparsesign`, `complex_gaussian`, `complex_srtt` are supported
- `sketch_size=-1`: Sketch dimension s (default: `max(5*jmax, 5*k)`)
- `orth_method=:rcgs`: Θ-orthogonalization method (`:rcgs`, `:rcgs2`, `:rqr`)
- `sketch_prec=Float64`: real floating-point type for sketch arrays; Float32 is not
  recommended as Mc is computed from the sketch and requires full precision

# Returns
`(X, lambda, history)` where X is n×k, lambda is length k,
and history is nit×3 with columns [max_rnorm, iter, nmv].
"""
@views @timing "jd_sketched_standard" function jd_sketched_standard(A, v0::AbstractArray;
                          k::Int=5,
                          tol::Float64=1e-8,
                          maxit::Int=200,
                          M=nothing,
                          precond_preparator=nothing,
                          disp::Bool=false,
                          sketch_type::String="sparsestack",
                          sketch_size::Int=-1,
                          orth_method::Symbol=:rcgs,
                          sketch_prec::Type{<:AbstractFloat}=Float64)

    if eltype(v0) <: Real && sketch_type in ("real_gaussian", "real_srtt")
        throw(ArgumentError(
            "sketch_type=\"$sketch_type\" produces a real sketch matrix, which is " *
            "incompatible with jd_sketched_standard on real inputs (requires complex sketch). " *
            "Use sparsestack, sparsesign, complex_gaussian, or complex_srtt."))
    end

    n = size(A, 1)
    k = min(k, n)
    nbuff = 2
    jbuff = floor(Int, max(4, 0.2 * k))
    kb    = min(k + nbuff, n)
    jmin  = min(n, 2 * (k + nbuff)) + jbuff
    jmax  = min(n, 4 * kb) + jbuff
    s     = sketch_size < 0 ? max(5 * jmax, 5 * k) : sketch_size

    # Force complex so that Mc = SV^H SW is approximately Hermitian (not just symmetric).
    # This makes eigen(Mc) well-conditioned and avoids complex-eigenvalue sorting issues.
    T  = complex(eltype(v0))
    TS = complex(sketch_prec)

    # Pass a complex-type template so that sparsestack/sparsesign build a ComplexF32 sketch
    # matrix, whose MatrixSketchOp buffer can hold ComplexF64 inputs (truncated to ComplexF32).
    Theta = sketch(n, s, sketch_type, similar(v0, T, 1); prec=sketch_prec)

    # ── Workspace ─────────────────────────────────────────────────────────
    @timing "jd_sketched_standard: allocation" begin
        V  = similar(v0, T,  n, jmax)
        W  = similar(v0, T,  n, jmax)
        SV = fill!(similar(v0, TS, s, jmax), zero(TS))
        SW = fill!(similar(v0, TS, s, jmax), zero(TS))
        n_buffer = similar(v0, T,  n, max(2*kb, jmin))
        s_buffer = similar(v0, TS, s, max(2*kb, jmin))
    end

    nconv    = 0
    nmv      = 0
    history  = zeros(Float64, maxit, 3)
    hist_row = 0

    # ── Initialize subspace ───────────────────────────────────────────────
    j  = min(kb, n)
    nc = min(size(v0, 2), j)
    V[:, 1:nc] .= v0[:, 1:nc]
    if nc < j
        randn!(TaskLocalRNG(), V[:, nc+1:j])
    end

    @timing "jd_sketched_standard: sketch" mul!(SV[:, 1:j], Theta, V[:, 1:j])
    j = _sketch_ortho!(V[:, 1:j], SV[:, 1:j], orth_method, n_buffer[:, 1:j], s_buffer[:, 1:j])

    @timing "jd_sketched_standard: matvec" begin
        mul!(W[:, 1:j], A, V[:, 1:j])
        nmv += j
    end
    @timing "jd_sketched_standard: sketch" mul!(SW[:, 1:j], Theta, W[:, 1:j])

    # Mc = (ΘV)ᴴ(ΘAV) — approximately Hermitian for complex V and Hermitian A
    Mc = SV[:, 1:j]' * SW[:, 1:j]

    @timing "jd_sketched_standard: diag" begin
        F    = eigen(Mc)
        ew   = real.(F.values)
        U    = F.vectors
        perm = sortperm(ew)
        ew   = copy(ew[perm])
        U    = copy(U[:, perm])
    end

    # ── Main loop ──────────────────────────────────────────────────────────
    jd_iter = 0
    for iter in 1:maxit
        jd_iter = iter

        nb = max(min(k - nconv + nbuff, j - nconv), 1)

        # Restart: U from general eigen is non-unitary → QR-orthonormalize
        if j + nb >= jmax
            @timing "jd_sketched_standard: restart" begin
                nk    = min(jmin, j)
                # qr of the ComplexF32 eigenvectors; result is ComplexF32
                Q_rst = Matrix(qr(U[:, 1:nk]).Q)
                mul!(n_buffer[:, 1:nk], V[:, 1:j],  T.(Q_rst))
                V[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SV[:, 1:j], Q_rst)
                SV[:, 1:nk] .= s_buffer[:, 1:nk]
                mul!(n_buffer[:, 1:nk], W[:, 1:j],  T.(Q_rst))
                W[:, 1:nk] .= n_buffer[:, 1:nk]
                mul!(s_buffer[:, 1:nk], SW[:, 1:j], Q_rst)
                SW[:, 1:nk] .= s_buffer[:, 1:nk]
                j  = nk
                nb = max(min(k - nconv + nbuff, j - nconv), 1)
                Mc    = Q_rst' * Mc * Q_rst
                F_rst = eigen(Mc)
                ew    = real.(F_rst.values)
                U     = F_rst.vectors
                perm  = sortperm(ew)
                ew    = copy(ew[perm])
                U     = copy(U[:, perm])
                disp && @printf("  RESTART -> j=%d  nconv=%d/%d\n", j, nconv, k)
            end
        end

        # Ritz vectors and residuals
        Y_nb = view(U, :, nconv+1:nconv+nb)
        ub   = view(n_buffer, :, 1:nb)
        rb   = view(n_buffer, :, nb+1:2*nb)
        mul!(ub, V[:, 1:j], T.(Y_nb))
        mul!(rb, W[:, 1:j], T.(Y_nb))

        # Sketched Rayleigh quotients for residual computation
        rnorms = let SUb = view(s_buffer, :, 1:nb), SWb = view(s_buffer, :, nb+1:2*nb)
            mul!(SUb, SV[:, 1:j], Y_nb)
            mul!(SWb, SW[:, 1:j], Y_nb)
            theta_b = T.(real.(columnwise_dots(SUb, SWb) ./ columnwise_dots(SUb, SUb)))
            rb .-= ub .* theta_b'
            @timing "jd_sketched_standard: sketch" mul!(SWb, Theta, rb)
            columnwise_norms(SWb)
        end

        nb_target = min(k - nconv, nb)
        hist_row += 1
        history[hist_row, :] .= (maximum(rnorms[1:nb_target]), iter, nmv)

        if disp
            @printf("jd_sketched_standard it=%4d  nconv=%d/%d  j=%d  nb=%d  |r|=%.2e..%.2e\n",
                    iter, nconv, k, j, nb,
                    minimum(rnorms[1:nb_target]), maximum(rnorms[1:nb_target]))
        end

        nconv_new = 0
        if iter > 1
            first_notconv = findfirst(rnorms[1:nb_target] .>= tol)
            nconv_new = isnothing(first_notconv) ? nb_target : first_notconv - 1
        end

        if nconv_new > 0
            nconv += nconv_new
            nconv >= k && break
            nb -= nconv_new
            if nb > 0
                rb[:, 1:nb] .= rb[:, nconv_new+1:nconv_new+nb]
                ub[:, 1:nb] .= ub[:, nconv_new+1:nconv_new+nb]
            else
                continue
            end
        end

        if !isnothing(M)
            !isnothing(precond_preparator) && precond_preparator(M, ub[:, 1:nb])
            ldiv!(ub[:, 1:nb], M, rb[:, 1:nb])
            rb[:, 1:nb] .= ub[:, 1:nb]
        end

        SV_scratch = view(s_buffer, :, 1:nb)
        V_scratch  = view(n_buffer, :, 1:nb)
        S_scratch  = view(s_buffer, :, nb+1:2*nb)
        Mc, nact = _jdrb_expand_standard!(V, SV, W, SW, Mc, rb[:, 1:nb], j, nb, jmax, A,
                                          Theta, orth_method, V_scratch, SV_scratch, S_scratch)
        nmv += nact
        j   += nact

        @timing "jd_sketched_standard: diag" begin
            F    = eigen(Mc)
            ew   = real.(F.values)
            U    = F.vectors
            perm = sortperm(ew)
            ew   = copy(ew[perm])
            U    = copy(U[:, perm])
        end
    end

    disp && @printf("jd_sketched_standard: done. nconv=%d/%d  iter=%d  nmv=%d\n",
                    nconv, k, jd_iter, nmv)
    if nconv < k
        @warn "jd_sketched_standard did not converge within $maxit iterations."
    end

    # Finalize: V is Θ-orthonormal (not standard-orthonormal), U is non-unitary.
    # Rotate to best k Ritz vectors, Cholesky re-orthonormalize, solve small eigenproblem.
    @timing "jd_sketched_standard: finalize" begin
        F_fin  = eigen(Mc)
        ew_fin = real.(F_fin.values)
        U_fin  = F_fin.vectors
        perm_f = sortperm(ew_fin)
        U_fin  = T.(copy(U_fin[:, perm_f[1:k]]))
        mul!(n_buffer[:, 1:k], V[:, 1:j], U_fin)
        V[:, 1:k] .= n_buffer[:, 1:k]
        mul!(n_buffer[:, 1:k], W[:, 1:j], U_fin)
        W[:, 1:k] .= n_buffer[:, 1:k]
        Xritz = view(V, :, 1:k)
        Writz = view(W, :, 1:k)
        R = cholesky(Hermitian(Xritz' * Xritz)).U
        rdiv!(Xritz, R)
        rdiv!(Writz, R)
        Fk     = eigen(Hermitian(Xritz' * Writz))
        X_full = Xritz * Fk.vectors
        lambda = Fk.values
        perm_s = sortperm(lambda)
        X_sorted = copy(X_full[:, perm_s])
        lambda   = real.(copy(lambda[perm_s]))
        if eltype(v0) <: Real
            # Remove per-column complex phase (from complex arithmetic) then take real part.
            absvals = abs.(X_sorted)
            maxabs, indices = findmax(absvals; dims=1)
            X_sorted ./= X_sorted[indices] ./ maxabs
            X_out = real.(X_sorted)
        else
            X_out = X_sorted
        end
    end

    return X_out, lambda, history[1:hist_row, :]
end


# ── Helper ────────────────────────────────────────────────────────────────

"""
Expand search subspace for the standard (non-generalized) sketched JD.

Mirrors `_jdrb_expand!` from `jd_sketched.jl` but maintains both SV = ΘV and
SW = ΘAV and returns the expanded non-symmetric Mc = SV'SW.
"""
@views function _jdrb_expand_standard!(V, SV, W, SW, Mc, rb, j, nb, jmax, A, Theta,
                                       orth_method, V_scratch, SV_scratch, S_scratch)
    @timing "jd_sketched_standard: sketch" mul!(SV_scratch[:, 1:nb], Theta, rb[:, 1:nb])
    @timing "jd_sketched_standard: ortho" begin
        _sketch_project_out!(rb[:, 1:nb], SV_scratch[:, 1:nb], V[:, 1:j], SV[:, 1:j], orth_method)
        nact = _sketch_ortho!(rb[:, 1:nb], SV_scratch[:, 1:nb], orth_method,
                              V_scratch[:, 1:nb], S_scratch[:, 1:nb])
    end

    nact == 0 && return Mc, 0
    nact = min(nact, jmax - j)
    nact == 0 && return Mc, 0

    V[:,  j+1:j+nact] .= rb[:,          1:nact]
    SV[:, j+1:j+nact] .= SV_scratch[:,  1:nact]

    @timing "jd_sketched_standard: matvec" mul!(W[:, j+1:j+nact], A, V[:, j+1:j+nact])
    @timing "jd_sketched_standard: sketch" mul!(SW[:, j+1:j+nact], Theta, W[:, j+1:j+nact])

    @timing "jd_sketched_standard: overlap" begin
        SW_new = view(SW, :, j+1:j+nact)
        SV_new = view(SV, :, j+1:j+nact)
        Mexp   = similar(Mc, j+nact, j+nact)
        Mexp[1:j,        1:j]        .= Mc
        mul!(Mexp[1:j,        j+1:j+nact], SV[:, 1:j]', SW_new)
        mul!(Mexp[j+1:j+nact, 1:j],        SV_new',      SW[:, 1:j])
        mul!(Mexp[j+1:j+nact, j+1:j+nact], SV_new',      SW_new)
    end

    return Mexp, nact
end
