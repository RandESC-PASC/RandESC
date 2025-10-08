# ira.jl
using LinearAlgebra, Random

"""
    ira(A, k, m; which="SR", maxit=100, tol=1e-8, v0=nothing,
                 orth_method="cgs2", verbose=true)

Implicitly Restarted Arnoldi for k Ritz pairs from an n×n operator `A`
(matrix or function x->A*x). Arnoldi subspace dimension m must satisfy m>k.

Returns `(V, D, ritz, info)` where:
- `V :: Matrix`  — n×k Ritz vectors (columns)
- `D :: Diagonal`— k×k diagonal of Ritz values
- `ritz :: NamedTuple` — `(theta, res)` final length-m Ritz values & residual estimates
- `info :: NamedTuple` — `(iters, mvps, converged)`
"""
function ira(A, k::Integer; m::Integer = max(2*k,k+20),
             which::AbstractString="SR", maxit::Integer=100, tol::Real=1e-8,
             v0::Union{Nothing,AbstractVector}=nothing,
             orth_method::AbstractString="cgs2", verbose::Bool=true)

    n, v0 = infer_n_and_v0(A, v0)
    if m <= k
        error("Require m > k (got m=$m, k=$k).")
    end
    if !(1 ≤ k ≤ n-1)
        error("k must be in [1, n-1] (got k=$k, n=$n).")
    end

    # Initial Arnoldi to size m
    V_m, H_m, beta, f, mvps = arnoldi(A, v0, m, orth_method)

    iters = 0
    converged = 0
    Y = nothing; theta = nothing; res = nothing

    while iters < maxit
        iters += 1

        # Small eig on H_m
        F = ishermitian(H_m) ? eigen(Hermitian(H_m)) : eigen(H_m)
        theta = F.values
        Y     = F.vectors

        ord = sort_which(theta, which)
        theta = theta[ord]
        Y     = Y[:, ord]

        # Arnoldi residual estimates: ||r_i|| = beta * |e_m^T y_i| = beta * |y_i(end)|
        res = abs.(beta .* vec(Y[end, :]))

        if all(res[1:k] .<= tol)
            converged = k
            break
        else
            converged = count(≤(tol), res[1:k])
        end

        # ----- IRA restart: apply p = m-k implicit QR shifts (unwanted Ritz values)
        p = m - k
        shifts = theta[k+1 : min(m, k+p)]

        H = H_m
        for mu in shifts
            Fqr = qr(H - mu*I)               # shifted QR
            Q   = Matrix(Fqr.Q)
            R   = Matrix(Fqr.R)
            H   = R*Q + mu*I                 # similarity step
            V_m = V_m * Q                     # accumulate basis rotation
        end

        # Restart from first k columns; expand back to m
        Vk0 = V_m[:, 1:k]
        H_k = H[1:k, 1:k]
        V_m, H_m, beta, f, mvps_add = arnoldi_expand_from_block(A, Vk0, H_k, m, orth_method)
        mvps += mvps_add

        if verbose
            println("IRA iter $(iters): max(res[1..k])=$(maximum(res[1:k]))  mvps=$(mvps)")
        end
    end

    # Final extraction on last H_m
    Yk = Y[:, 1:k]
    Vk = V_m * Yk

    V = Vk
    D = Diagonal(theta[1:k])
    ritz = (theta = theta, res = res)
    info = (it = iters, mvps = mvps, res = res)
    return V, D, ritz, info
end

# --------------------- Arnoldi building blocks -----------------------------

# Standard Arnoldi from v1 to dimension m
function arnoldi(A, v1::AbstractVector, m::Integer, orth_method::AbstractString)
    n = length(v1)
    T = ComplexF64  # safe universal element type (handles real/complex)
    V = zeros(T, n, m)
    H = zeros(T, m, m)
    mvps = 0

    V[:, 1] = v1 / norm(v1)

    for j in 1:m-1
        w = applyA(A, V[:, j]); mvps += 1
        w, hcol = orth_step(view(V, :, 1:j), w, orth_method)
        H[1:j, j] .= hcol
        H[j+1, j] = norm(w)
        if H[j+1, j] == 0
            V = V[:, 1:j]; H = H[1:j, 1:j]; f = zeros(T, n); beta = zero(T)
            return V, H, beta, f, mvps
        end
        V[:, j+1] .= w / H[j+1, j]
    end

    w = applyA(A, V[:, m]); mvps += 1
    w, hcol = orth_step(view(V, :, 1:m), w, orth_method)
    H[1:m, m] .= hcol
    beta = norm(w)
    f = w
    return V, H, beta, f, mvps
end

# Restart/expand Arnoldi from a k-block to dimension m
function arnoldi_expand_from_block(A, V0::AbstractMatrix, H_k::AbstractMatrix, m::Integer,
                                   orth_method::AbstractString)
    n, k = size(V0)
    T = eltype(V0) <: Complex ? eltype(V0) : ComplexF64
    V = zeros(T, n, m)
    H = zeros(T, m, m)
    V[:, 1:k] .= V0
    H[1:k, 1:k] .= H_k
    mvps = 0

    for j in k:(m-1)
        w = applyA(A, V[:, j]); mvps += 1
        w, hcol = orth_step(view(V, :, 1:j), w, orth_method)
        H[1:j, j] .= hcol
        H[j+1, j] = norm(w)
        if H[j+1, j] == 0
            V = V[:, 1:j]; H = H[1:j, 1:j]; beta = zero(T); f = zeros(T, n)
            return V, H, beta, f, mvps
        end
        V[:, j+1] .= w / H[j+1, j]
    end

    w = applyA(A, V[:, m]); mvps += 1
    w, hcol = orth_step(view(V, :, 1:m), w, orth_method)
    H[1:m, m] .= hcol
    beta = norm(w)
    f = w
    return V, H, beta, f, mvps
end

# --------------------- Orthonormalization options --------------------------

# Orthonormalize w against columns of V (n×j).
#   'mgs'  : Modified Gram–Schmidt (1 pass)
#   'mgs2' : MGS with reorthogonalization (DGKS)  **recommended**
#   'cgs'  : Classical GS (1 pass)
#   'cgs2' : CGS with reorth
function orth_step(V::AbstractMatrix, w::AbstractVector, orth_method::AbstractString)
    j = size(V, 2)
    j == 0 && return w, zeros(eltype(w), 0)

    meth = lowercase(orth_method)
    if meth in ("mgs", "mgs1")
        hcol = zeros(eltype(w), j)
        @inbounds for i in 1:j
            h = V[:, i]' * w
            w -= V[:, i] * h
            hcol[i] = h
        end
        return w, hcol

    elseif meth in ("mgs2", "dgks")
        hcol = zeros(eltype(w), j)
        @inbounds for i in 1:j
            h = V[:, i]' * w
            w -= V[:, i] * h
            hcol[i] = h
        end
        @inbounds for i in 1:j
            h = V[:, i]' * w
            w -= V[:, i] * h
            hcol[i] += h
        end
        return w, hcol

    elseif meth == "cgs"
        hcol = V' * w
        w   -= V * hcol
        return w, hcol

    elseif meth == "cgs2"
        hcol = V' * w
        w   -= V * hcol
        h2   = V' * w
        w   -= V * h2
        hcol += h2
        return w, hcol

    else
        error("Unknown orth_method \"$orth_method\".")
    end
end

# --------------------------- Utilities -------------------------------------

applyA(A::AbstractMatrix, x::AbstractVector) = A * x
applyA(A::Function,      x::AbstractVector) = A(x)

function infer_n_and_v0(A, v0)
    if v0 !== nothing
        v = vec(v0)
        n = length(v)
    else
        if A isa AbstractMatrix
            n = size(A, 1)
            v = randn(n)
        else
            error("For function A, please pass v0 to infer n.")
        end
    end
    v = v ./ norm(v)
    return n, v
end

# Return permutation that orders `theta` according to `which`
function sort_which(theta::AbstractVector, which::AbstractString)
    W = uppercase(which)
    if W == "LM"
        return sortperm(abs.(theta), rev=true)
    elseif W == "SM"
        return sortperm(abs.(theta), rev=false)
    elseif W == "LR"
        return sortperm(real.(theta), rev=true)
    elseif W == "SR"
        return sortperm(real.(theta), rev=false)
    elseif W == "LI"
        return sortperm(abs.(imag.(theta)), rev=true)
    elseif W == "SI"
        return sortperm(abs.(imag.(theta)), rev=false)
    else
        error("Unknown which = $which")
    end
end
