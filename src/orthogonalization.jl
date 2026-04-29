using LinearAlgebra

# === Standard orthogonalization utilities ===

"""
Orthonormalize V[:,1:m] in-place using the chosen method. Linearly dependent columns
are removed: independent columns are compacted to V[:,1:nact]. Returns nact.
  :mgs  — single-pass modified Gram-Schmidt
  :mgs2 — double-pass modified Gram-Schmidt (more accurate, more expensive)
  :qr   — Householder QR (batch)
buf (nx≥m) is a workspace for :qr; ignored otherwise.
"""
function _jd_ortho!(V::AbstractMatrix, m::Int, orth_method::Symbol,
                    buf::AbstractMatrix=similar(V, size(V, 1), m))
    if orth_method == :qr
        return _jd_qr!(V, m, buf)
    else
        return _jdb_mgs2!(V, m, orth_method == :mgs2)
    end
end

"""Householder QR orthonormalization of V[:,1:m] in-place. Compacts linearly independent
columns to the front and returns their count."""
@views function _jd_qr!(V::Matrix, m::Int, buf::Matrix)
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

@views function _jd_qr!(V::AbstractMatrix, m::Int, buf::AbstractMatrix)
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

"""MGS orthonormalization of V[:,1:m] in-place. Compacts linearly independent columns
to the front and returns their count."""
function _jdb_mgs2!(V::AbstractMatrix, m::Int, two_pass::Bool=true)
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

"""Sketched QR orthonormalization of V0 (nxm) in-place.
Writes nact accepted columns into V_out[:,1:nact] and SV_out[:,1:nact].
SV_buf (s×m) is scratch for the sketch. Returns nact."""
@views function _jd_theta_qr!(V_out::AbstractMatrix, SV_out::AbstractMatrix,
                               V0::AbstractMatrix, Theta, SV_buf::AbstractMatrix;
                               tol::Float64=1e-10)
    m = size(V0, 2)
    SV = view(SV_buf, :, 1:m)
    mul!(SV, Theta, V0)
    F = qr(SV)
    good = findall(abs.(F.R[diagind(F.R)]) .>= tol)
    nact = length(good)
    fill!(SV_out[:, 1:m], zero(eltype(SV_out)))
    SV_out[diagind(SV_out)[1:m]] .= one(eltype(SV_out))
    lmul!(F.Q, SV_out[:, 1:m])
    if nact < m
        SV_out[:, 1:nact] .= SV_out[:, good]
    end
    # QR on sketch: Θ V0 = Q_S R, so Θ V0[:,good] = Q_S[:,1:nact] * R[good,good].
    # Right-multiplying by R[good,good]^-1: Θ (V0[:,good] / R[good,good]) = Q_S[:,1:nact].
    # So V0 * R^-1 is sketch orthogonal
    V_out[:, 1:nact] .= V0[:, good]
    rdiv!(V_out[:, 1:nact], UpperTriangular(F.R[good, good]))
    return nact
end

"""
    theta_orth_block_against!(V0, Q, SQ, Theta, method, SV_buf)

In-place Θ-orthogonalization of `V0` against Θ-orthonormal `Q`.
`SV_buf` (s × size(V0,2)) is a pre-allocated scratch array.
"""
function theta_orth_block_against!(V0::AbstractMatrix, Q, SQ, Theta, method::Symbol,
                                   SV_buf::AbstractMatrix)
    isempty(Q) && return V0
    nb = size(V0, 2)
    SV = view(SV_buf, :, 1:nb)
    mul!(SV, Theta, V0)
    H = SQ' * SV
    mul!(V0, Q, H, -1, 1)
    if method == :rcgs2
        mul!(SV, Theta, V0)
        mul!(H, SQ', SV)
        mul!(V0, Q, H, -1, 1)
    end
    return V0
end


"""
    theta_orth_block!(V_out, SV_out, V0, Theta, method, SV_buf) -> nact

In-place Θ-orthonormalization. Writes accepted vectors into the first `nact` columns of
pre-allocated `V_out` (n×p) and `SV_out` (s×p). `SV_buf` (s×p) is scratch. Returns `nact`.
"""
@views function theta_orth_block!(V_out::AbstractMatrix, SV_out::AbstractMatrix,
                                  V0::AbstractMatrix, Theta, method::Symbol,
                                  SV_buf::AbstractMatrix; tol::Float64=1e-12)
    _, p = size(V0)
    mul!(SV_buf[:, 1:p], Theta, V0)
    ncols = 0
    npass = (method == :rcgs2) ? 2 : 1

    for i in 1:p
        v  = view(V0,     :, i)
        sv = view(SV_buf, :, i)

        if ncols > 0
            Vc  = view(V_out,  :, 1:ncols)
            SVc = view(SV_out, :, 1:ncols)
            for _ in 1:npass
                h = SVc' * sv
                mul!(v,  Vc,  h, -1, 1)
                mul!(sv, SVc, h, -1, 1)
            end
        end

        nv = norm(sv)
        if nv > tol
            ncols += 1
            V_out[:,  ncols] .= v  ./ nv
            SV_out[:, ncols] .= sv ./ nv
        end
    end

    return ncols
end

function _jd_theta_ortho!(V_out::AbstractMatrix, SV_out::AbstractMatrix,
                           V0::AbstractMatrix, Theta, method::Symbol,
                           SV_buf::AbstractMatrix)
    if method == :rqr
        return _jd_theta_qr!(V_out, SV_out, V0, Theta, SV_buf)
    else
        return theta_orth_block!(V_out, SV_out, V0, Theta, method, SV_buf)
    end
end
