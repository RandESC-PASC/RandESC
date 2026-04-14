using LinearAlgebra

# === Sketched / Θ-orthogonalization utilities ===

function init_space_sketched(n::Int, v0, Theta, X_converged, SX_converged, T::Type,
                             orth_method::Symbol, jmax::Int=100)
    """Initialize search space with Θ-orthonormal vectors.

    If v0 is a matrix, uses all columns (up to jmax) as initial vectors.
    If v0 is a vector or nothing, uses a single random vector.
    """

    # Determine number of initial vectors
    if isnothing(v0)
        V0 = randn(T, n, 1)
    elseif ndims(v0) == 1
        # Single vector input
        V0 = reshape(T.(v0), n, 1)
    else
        # Matrix input - use all columns up to jmax
        p = min(size(v0, 2), jmax)
        V0 = T.(v0[:, 1:p])
    end

    # Θ-orthogonalize against converged space first
    if !isempty(X_converged)
        V0 = theta_orth_block_against(V0, X_converged, SX_converged, Theta, orth_method)
    end

    # Θ-orthonormalize the block internally
    V, SV = theta_orth_block(V0, Theta, orth_method)

    # Check if we got any valid vectors; if not, generate random ones
    if size(V, 2) == 0
        V0 = randn(T, n, 1)
        if !isempty(X_converged)
            V0 = theta_orth_block_against(V0, X_converged, SX_converged, Theta, orth_method)
        end
        V, SV = theta_orth_block(V0, Theta, orth_method)
    end

    return V, SV
end


function theta_orth_block(V0, Theta, method::Symbol; tol::Float64=1e-14)
    """Θ-orthonormalize a block of vectors.

    Returns V, SV where SV = Θ*V has ℓ₂-orthonormal columns.
    Vectors with Θ-norm below tol are discarded.

    Methods:
      :rcgs  - Randomized CGS (single pass)
      :rcgs2 - Randomized CGS with reorthogonalization (two passes)
      :rgs   - Randomized GS via least squares (default)
    """
    n, p = size(V0)
    T = eltype(V0)

    if p == 0
        s = size(Theta(zeros(T, n)), 1)
        return zeros(T, n, 0), zeros(T, s, 0)
    end

    # Sketch entire block at once (one BLAS call, not p separate calls)
    SV0 = Theta(V0)
    s = size(SV0, 1)

    # Pre-allocate output; use ncols to track accepted vectors
    V  = zeros(T, n, p)
    SV = zeros(T, s, p)
    ncols = 0

    for i in 1:p
        v  = V0[:,  i]
        sv = SV0[:, i]

        if ncols > 0
            Vc  = view(V,  :, 1:ncols)
            SVc = view(SV, :, 1:ncols)

            if method == :rgs
                h  = SVc \ sv
                v  = v  - Vc  * h
                sv = sv - SVc * h
            else  # :rcgs / :rcgs2
                npass = (method == :rcgs2) ? 2 : 1
                for _ in 1:npass
                    h  = SVc' * sv
                    v  = v  - Vc  * h
                    sv = sv - SVc * h
                end
            end
        end

        nv = norm(sv)
        if nv > tol
            ncols += 1
            V[:,  ncols] .= v  ./ nv
            SV[:, ncols] .= sv ./ nv
        end
    end

    return V[:, 1:ncols], SV[:, 1:ncols]
end


function theta_orth_block_against(V0, Q, SQ, Theta, method::Symbol)
    """Θ-orthogonalize block V0 against existing Θ-orthonormal block Q.

    Methods:
      :rcgs  - Randomized CGS (single pass)
      :rcgs2 - Randomized CGS with reorthogonalization (two passes)
      :rgs   - Randomized GS via least squares (default)
    """
    if isempty(Q)
        return V0
    end

    # Compute sketch of V0 directly (no copy of V0 needed)
    SV = Theta(V0)

    if method == :rcgs
        H = SQ' * SV
        return V0 - Q * H

    elseif method == :rcgs2
        V = V0 - Q * (SQ' * SV)
        SV = Theta(V)
        H2 = SQ' * SV
        return V - Q * H2

    else  # :rgs (default)
        H = SQ \ SV
        return V0 - Q * H
    end
end


function theta_orth_against(v, V, SV, Theta, method::Symbol)
    """Θ-orthogonalize v against V using specified method

    Methods:
      :rcgs  - Randomized CGS (single pass)
      :rcgs2 - Randomized CGS with reorthogonalization (two passes)
      :rgs   - Randomized GS via least squares (default)
    """
    if isempty(V)
        return v
    end
    v = copy(v)

    if method == :rcgs
        # Randomized CGS (single pass)
        sv = Theta(v)
        h = SV' * sv
        v = v - V * h

    elseif method == :rcgs2
        # Randomized CGS with reorthogonalization (two passes)
        for _ in 1:2
            sv = Theta(v)
            h = SV' * sv
            v = v - V * h
        end

    else  # :rgs (default)
        # Randomized GS via least squares
        sv = Theta(v)
        h = SV \ sv
        v = v - V * h
    end

    return v
end


"""
    theta_orth_block_against!(V0, Q, SQ, Theta, method, SV_buf)

In-place version of `theta_orth_block_against`. Modifies `V0` in-place.
`SV_buf` (s × size(V0,2)) is a pre-allocated scratch array.

Since Q is Θ-orthonormal, SQ has orthonormal ℓ₂ columns (SQ'*SQ = I), so projection
coefficients are H = SQ'*SV for all methods — `\\` and `'*` are equivalent here, but
`mul!` avoids the QR factorization that `\\` would trigger internally.
The method argument controls the number of passes only.
"""
function theta_orth_block_against!(V0::AbstractMatrix, Q, SQ, Theta, method::Symbol,
                                   SV_buf::AbstractMatrix)
    isempty(Q) && return V0
    nb = size(V0, 2)
    nq = size(SQ, 2)
    SV = view(SV_buf, :, 1:nb)
    mul!(SV, Theta, V0)           # in-place sketch, no allocation
    H = SQ' * SV                  # projection coefficients, no allocation
    mul!(V0, Q,   H, -1, 1)       # V0 -= Q*H, in-place
    if method == :rcgs2
        mul!(SV, Theta, V0)       # re-sketch in-place
        mul!(H,  SQ', SV)
        mul!(V0, Q,   H, -1, 1)
    end
    return V0
end


"""
    theta_orth_block!(V_out, SV_out, V0, Theta, method, SV_buf) -> nact

In-place version of `theta_orth_block`. Writes accepted vectors into the first `nact`
columns of the pre-allocated `V_out` (n×p) and `SV_out` (s×p).
`SV_buf` (s×p) is a pre-allocated scratch array.
Returns `nact`, the number of accepted (non-degenerate) vectors.

Since each accepted column is Θ-normalized, SVc has orthonormal ℓ₂ columns, so
h = SVc' * sv is exact (same result as SVc\\sv_work but without the QR allocation).
The method argument controls the number of orthogonalization passes.
"""
@views function theta_orth_block!(V_out::AbstractMatrix, SV_out::AbstractMatrix,
                                  V0::AbstractMatrix, Theta, method::Symbol,
                                  SV_buf::AbstractMatrix; tol::Float64=1e-14)
    n, p = size(V0)
    mul!(SV_buf[:, 1:p], Theta, V0)         # sketch entire input block in-place
    ncols = 0
    npass = (method == :rcgs2) ? 2 : 1      # :rgs and :rcgs both use 1 pass

    for i in 1:p
        v  = view(V0,     :, i)
        sv = view(SV_buf, :, i)

        if ncols > 0
            Vc  = view(V_out,  :, 1:ncols)
            SVc = view(SV_out, :, 1:ncols)
            for _ in 1:npass
                h = SVc' * sv
                mul!(v,  Vc,   h, -1, 1)
                mul!(sv, SVc,  h, -1, 1)
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


function solve_correction_sketched(Q, SQ, Theta, r, M, orth_method::Symbol,
                                   precond_preparator, u)
    """Olsen-style correction with Θ inner product"""
    t = copy(r)
    if !isnothing(M)
        # Prepare preconditioner if needed
        if !isnothing(precond_preparator)
            precond_preparator(M, u)
        end
        t = M \ t
    end
    # Project out Q using Θ inner product
    t = theta_orth_against(t, Q, SQ, Theta, orth_method)
    return t
end
