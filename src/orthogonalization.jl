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
