using LinearAlgebra
# Valid orthogonalization method symbols. All dispatch functions in this file validate
# their `method` argument against these tuples.
const STD_ORTH_METHODS    = (:mgs, :mgs2, :qr, :cholqr2, :cholqr3)
const SKETCH_ORTH_METHODS = (:rcgs, :rcgs2, :rqr, :cholqr2, :cholqr3)

# === Standard orthogonalization utilities ===

"""
Orthonormalize V[:,1:m] in-place using the chosen method. Linearly dependent columns
are removed: independent columns are compacted to V[:,1:nact]. Returns nact.
  :mgs     — single-pass modified Gram-Schmidt
  :mgs2    — double-pass modified Gram-Schmidt (more accurate, more expensive)
  :qr      — Householder QR (batch)
  :cholqr2 — Cholesky QR (double pass; falls back to :qr if Cholesky fails)
  :cholqr3 — Cholesky QR (triple pass; falls back to :qr if Cholesky fails)
buf (nx≥m) is a workspace for :qr; ignored otherwise.
"""
function _ortho!(V::AbstractMatrix, m::Int, orth_method::Symbol,
                 buf::AbstractMatrix=similar(V, size(V, 1), m))
    orth_method in STD_ORTH_METHODS ||
        error("Unknown orth_method :$orth_method; valid: $STD_ORTH_METHODS")
    if orth_method == :qr
        return _qr_ortho!(V, m, buf)
    elseif orth_method == :cholqr2
        return _cholqr_ortho!(V, m, buf; n_pass=2)
    elseif orth_method == :cholqr3
        return _cholqr_ortho!(V, m, buf; n_pass=3)
    elseif orth_method == :mgs
        return _mgs_ortho!(V, m, false)
    else  # :mgs2
        return _mgs_ortho!(V, m, true)
    end
end

"""
    _qr_ortho!(V, m, buf) -> nact

Householder QR orthonormalization of `V[:,1:m]` in-place. Compacts linearly independent
columns to the front and returns their count.

Two dispatch targets: the `Matrix` method calls LAPACK directly (`geqrf!`/`orgqr!`);
the `AbstractMatrix` fallback copies into `buf` and uses the generic `qr!`.
"""
@views function _qr_ortho!(V::Matrix, m::Int, buf::Matrix)
    tau    = view(buf, 1:m, 1)
    LAPACK.geqrf!(V[:, 1:m], tau)
    good = findall(abs.(V[diagind(V)[1:m]]) .>= 1e-14)
    nact = length(good)
    LAPACK.orgqr!(V[:, 1:m], tau, m)
    if nact < m
       V[:, 1:nact] .= V[:, good]
    end
    return nact
end

@views function _qr_ortho!(V::AbstractMatrix, m::Int, buf::AbstractMatrix)
    buf[:, 1:m] .= V[:, 1:m]                # copy; qr! overwrites input
    F = qr!(buf[:, 1:m])                    # in-place QR, no internal copy
    fill!(V[:, 1:m], zero(eltype(V)))
    V[diagind(V)[1:m]] .= one(eltype(V))    # identity in V[:,1:m]
    lmul!(F.Q, V[:, 1:m])                   # V[:,1:m] ← Q, no allocation
    good = findall(abs.(diag(F.R)) .>= 1e-14)
    nact = length(good)
    if nact < m
        V[:, 1:nact] .= V[:, good]
    end
    return nact
end

@views function _cholqr_ortho!(V::AbstractMatrix, m::Int, buf::AbstractMatrix;
                               n_pass::Int=2)
    Vm = V[:, 1:m]
    buf[:, 1:m] .= Vm

    for _ in 1:n_pass
        G = Hermitian(Vm' * Vm)

        F = try
            cholesky(G)
        catch
            V[:, 1:m] .= buf[:, 1:m]
            return _qr_ortho!(V, m, buf)
        end

        # Aggressive rank check, revert to QR if any diagonal entries are "near" zero or NaN
        if any(isnan, F.U) || minimum(abs, diag(F.U)) <= 1e-6
            V[:, 1:m] .= buf[:, 1:m]
            return _qr_ortho!(V, m, buf)
        end

        rdiv!(Vm, UpperTriangular(F.U))
    end

    return m
end

"""MGS orthonormalization of V[:,1:m] in-place. Compacts linearly independent columns
to the front and returns their count."""
function _mgs_ortho!(V::AbstractMatrix, m::Int, two_pass::Bool=true)
    nact = 0
    for i in 1:m
        vi = view(V, :, i)
        if nact > 0
            Vp = view(V, :, 1:nact)
            ci = Vp' * vi
            mul!(vi, Vp, ci, -1, 1)
            if two_pass
                mul!(ci, Vp', vi)
                mul!(vi, Vp, ci, -1, 1)
            end
        end
        nv = norm(vi)
        if nv > 1e-14
            vi ./= nv
            nact += 1
            if nact != i # avoid self copies
                V[:, nact] .= vi
            end
        end
    end
    return nact
end

"""
    _sketch_qr_ortho!(V, SV; tol) -> nact

Θ-orthonormalize columns of `V` (n×m) in-place via sketched QR, with pre-computed
sketch `SV` (s×m). Both are modified in-place; `nact` accepted columns are compacted
to `V[:,1:nact]` and `SV[:,1:nact]`.

`TV = eltype(V)` is the high-precision type (e.g. `Float64`);
`TS = eltype(SV)` is the low-precision sketch type (e.g. `Float32`).
Mixed-precision casts (e.g. `TV.(F.R)`) are applied only when `TV != TS`.
"""
@views function _sketch_qr_ortho!(V::AbstractMatrix, SV::AbstractMatrix;
                                  tol::Float64=1e-10)
    m = size(V, 2)
    F = qr(SV[:, 1:m])
    good = findall(abs.(F.R[diagind(F.R)]) .>= tol)
    nact = length(good)
    fill!(SV[:, 1:m], zero(eltype(SV)))
    SV[diagind(SV)[1:m]] .= one(eltype(SV))
    lmul!(F.Q, SV[:, 1:m])
    if nact < m
        SV[:, 1:nact] .= SV[:, good]
    end
    # QR on sketch: Θ V = Q_S R, so Θ (V R^-1) = Q_S.
    # rdiv! solves V[:,1:m] = V[:,1:m] * R^-1 in-place.
    TV = eltype(V)
    Rc = TV == eltype(F.R) ? F.R : TV.(F.R)
    rdiv!(V[:, 1:m], UpperTriangular(Rc))
    if nact < m
        V[:, 1:nact] .= V[:, good]
    end
    return nact
end

@views function _sketch_cholqr_ortho!(V::AbstractMatrix, SV::AbstractMatrix,
                                      V_buf::AbstractMatrix, S_buf::AbstractMatrix;
                                      n_pass::Int=2, tol::Float64=1e-10)
    m = size(V, 2)
    V_buf[:, 1:m] .= V
    S_buf[:, 1:m] .= SV

    for _ in 1:n_pass
        G = Hermitian(SV' * SV)

        F = try
            cholesky(G)
        catch
            V .= V_buf
            SV .= S_buf
            return _sketch_qr_ortho!(V, SV; tol=tol)
        end

        # Aggressive rank check, revert to QR if any diagonal entries are "near" zero or NaN
        if  any(isnan, F.U) || minimum(abs, diag(F.U)) <= 1e-6
            V .= V_buf
            SV .= S_buf
            return _sketch_qr_ortho!(V, SV; tol=tol)
        end

        U = UpperTriangular(F.U)
        rdiv!(V, U)
        rdiv!(SV, U)
    end

    return m
end

"""
    _sketch_project_out!(V0, SV0, Q, SQ, method)

Project out the Q-component from `V0` and its sketch `SV0` using Θ-inner products.
Both are updated in-place. `SV0` is maintained analytically (`SV0 -= SQ*(SQ'*SV0)`)
so no Θ application is needed — caller must pre-compute `SV0 = Θ*V0`.
"""
function _sketch_project_out!(V0::AbstractMatrix, SV0::AbstractMatrix, Q, SQ, method::Symbol)
    method in SKETCH_ORTH_METHODS ||
        error("Unknown sketch orth_method :$method; valid: $SKETCH_ORTH_METHODS")
    isempty(Q) && return
    TV = eltype(V0)
    H = SQ' * SV0
    mul!(V0,  Q,  TV == eltype(H) ? H : TV.(H), -1, 1)
    mul!(SV0, SQ, H, -1, 1)
    # :rcgs and :rqr both use a single projection pass before the subsequent
    # _sketch_ortho! step; only :rcgs2 repeats the projection here.
    if method == :rcgs2
        mul!(H,   SQ', SV0)
        mul!(V0,  Q,   TV == eltype(H) ? H : TV.(H), -1, 1)
        mul!(SV0, SQ,  H, -1, 1)
    end
end


"""
    _sketch_cgs_block!(V, SV, method) -> nact

Θ-orthonormalize columns of `V` (n×p) in-place, with pre-computed sketch `SV` (s×p).
Both `V` and `SV` are modified in-place; the `nact` accepted columns are compacted
to `V[:,1:nact]` and `SV[:,1:nact]`.
"""
@views function _sketch_cgs_block!(V::AbstractMatrix, SV::AbstractMatrix,
                                   method::Symbol; tol::Float64=1e-12)
    _, p = size(V)
    ncols = 0
    npass = method == :rcgs2 ? 2 : 1

    for i in 1:p
        v  = view(V,  :, i)
        sv = view(SV, :, i)

        if ncols > 0
            Vc  = view(V,  :, 1:ncols)
            SVc = view(SV, :, 1:ncols)
            TV  = eltype(v)
            for _ in 1:npass
                h = SVc' * sv
                mul!(v,  Vc,  TV == eltype(h) ? h : TV.(h), -1, 1)
                mul!(sv, SVc, h, -1, 1)
            end
        end

        nv = norm(sv)
        if nv > tol
            ncols += 1
            V[:,  ncols] .= v  ./ nv
            SV[:, ncols] .= sv ./ nv
        end
    end

    return ncols
end

"""
    _sketch_ortho!(V, SV, method, v_buf, s_buf) -> nact

Dispatch for all Θ-orthonormalization. `V` (n×p) and its pre-computed sketch `SV` (s×p)
are modified in-place; `nact` accepted columns are compacted to `V[:,1:nact]` / `SV[:,1:nact]`.

Throughout this file, `TV = eltype(V)` denotes the high-precision type (e.g. `Float64`)
and `TS = eltype(SV)` the low-precision sketch type (e.g. `Float32`). Mixed-precision
casts are applied only when `TV != TS`.

Valid `method` symbols:
  `:rcgs`  — single-pass randomized CGS
  `:rcgs2` — double-pass randomized CGS
  `:rqr`   — randomized QR via sketch; batch, preferred on GPU
  `:cholqr2` — Cholesky QR (double pass; falls back to :qr if Cholesky fails)
  `:cholqr3` — Cholesky QR (triple pass; falls back to :qr if Cholesky fails)
"""
function _sketch_ortho!(V::AbstractMatrix, SV::AbstractMatrix, method::Symbol,
                        v_buf::AbstractMatrix=similar(V), s_buf::AbstractMatrix=similar(SV))
    method in SKETCH_ORTH_METHODS ||
        error("Unknown sketch orth_method :$method; valid: $SKETCH_ORTH_METHODS")
    if method == :rqr
        return _sketch_qr_ortho!(V, SV)
    elseif method == :cholqr2
        return _sketch_cholqr_ortho!(V, SV, v_buf, s_buf; n_pass=2)
    elseif method == :cholqr3
        return _sketch_cholqr_ortho!(V, SV, v_buf, s_buf; n_pass=3)
    else  # :rcgs or :rcgs2
        return _sketch_cgs_block!(V, SV, method)
    end
end
