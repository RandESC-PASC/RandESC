using LinearAlgebra

# === Sketched / Θ-orthogonalization utilities ===

function init_space_sketched(n::Int, v0, Theta, X_converged, SX_converged, T::Type, orth_method::Symbol, jmax::Int=100)
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

    SV0 = Theta(V0)
    s = size(SV0, 1)

    if method == :rcgs || method == :rcgs2
        # Modified Gram-Schmidt in Θ-inner product
        npass = (method == :rcgs2) ? 2 : 1

        V = zeros(T, n, 0)
        SV = zeros(T, s, 0)

        for i in 1:p
            v = V0[:, i]
            sv = Theta(v)

            for _ in 1:npass
                if size(SV, 2) > 0
                    h = SV' * sv
                    v = v - V * h
                    sv = sv - SV * h
                end
            end

            nv = norm(sv)

            if nv > tol
                v = v / nv
                sv = sv / nv
                V = hcat(V, v)
                SV = hcat(SV, sv)
            end
        end

        return V, SV

    else  # :rgs (default) - Randomized GS via least squares
        V = zeros(T, n, 0)
        SV = zeros(T, s, 0)

        for i in 1:p
            v = V0[:, i]
            sv = Theta(v)

            if size(SV, 2) > 0
                h = SV \ sv
                v = v - V * h
                sv = sv - SV * h
            end

            nv = norm(sv)

            if nv > tol
                v = v / nv
                sv = sv / nv
                V = hcat(V, v)
                SV = hcat(SV, sv)
            end
        end

        return V, SV
    end
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

    V = copy(V0)
    SV = Theta(V)

    if method == :rcgs
        # Randomized block CGS (single pass)
        H = SQ' * SV
        V = V - Q * H

    elseif method == :rcgs2
        # Randomized block CGS with reorthogonalization (two passes)
        for _ in 1:2
            SV = Theta(V)
            H = SQ' * SV
            V = V - Q * H
        end

    else  # :rgs (default)
        # Randomized block GS via least squares
        H = SQ \ SV
        V = V - Q * H
    end

    return V
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


function solve_correction_sketched(Q, SQ, Theta, r, M, orth_method::Symbol, precond_preparator, u)
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
