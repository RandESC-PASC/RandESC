# rand_ira.jl
using LinearAlgebra, Random

include("sketch.jl")

"""
    rand_ira(A, k, m; which="SR", maxit=100, tol=1e-8, v0=nothing,
                      orth_method="rgs", verbose=true,
                      sketch_type="gaussian", sketch_s=nothing, sketch_seed=nothing)

Implicitly Restarted Arnoldi with *sketched* orthogonalization.

`A` may be an `AbstractMatrix` or a function `x -> A*x`.
Returns `(V, D, ritz, info)`:

- `V :: Matrix`         — n×k Ritz vectors
- `D :: Diagonal`       — diagonal Ritz values
- `ritz :: NamedTuple`  — `(theta, res)` final length-m Ritz values & sketched residuals
- `info :: NamedTuple`  — `(iters, mvps, converged)`
"""
function rand_ira(A, k::Integer, m::Integer;
                  which::AbstractString="SR", maxit::Integer=100, tol::Real=1e-8,
                  v0::Union{Nothing,AbstractVector}=nothing,
                  orth_method::AbstractString="rgs", verbose::Bool=true,
                  sketch_type::AbstractString="sparse",
                  sketch_s::Union{Nothing,Int}=nothing,
                  sketch_seed::Union{Nothing,Int}=nothing)

    n, v0 = infer_n_and_v0(A, v0)
    (m > k) || error("Require m > k (got m=$m, k=$k).")
    (1 ≤ k ≤ n-1) || error("k must be in [1, n-1] (got k=$k, n=$n).")

    s = isnothing(sketch_s) ? min(n, max(2*m, 6*k)) : sketch_s
    SS = sketch(n, s, sketch_type; seed=sketch_seed)

    mvps = 0; iters = 0

    # Initial Arnoldi (sketch-orthogonalized) to size m
    V_m, SV_m, H_m, beta, f, mvps_add = arnoldi_sketch(A, v0, m, SS, s, orth_method)
    mvps += mvps_add

    converged = 0
    Y = nothing; theta = nothing; res = nothing

    while iters < maxit
        iters += 1

        # Ritz pairs from small H_m
        F = eigen(H_m)                 # general (H_m is upper-Hessenberg)
        theta = F.values; Y = F.vectors

        ord = sort_which(theta, which)
        theta = theta[ord]; Y = Y[:, ord]

        # Sketched residual estimates: ||S r_i|| = beta * |e_m^T y_i|
        res = abs.(beta .* @view Y[end, :])

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
            Fqr = qr(H - mu*I)   # shifted QR (no broadcasting!)
            Q   = Matrix(Fqr.Q)
            R   = Matrix(Fqr.R)
            H   = R*Q + mu*I     # similarity update; preserves spectrum
            V_m = V_m * Q        # rotate bases
            SV_m = SV_m * Q
        end

        # Restart from first k columns; expand back to m
        Vk0  = V_m[:, 1:k]
        SVk0 = SV_m[:, 1:k]
        H_k  = H[1:k, 1:k]
        V_m, SV_m, H_m, beta, f, mvps_add = arnoldi_expand_from_block_sketch(
            A, Vk0, SVk0, H_k, m, SS, s, orth_method)
        mvps += mvps_add

        if verbose
            println("rand_ira iter $(iters): max(res[1..k])=$(maximum(res[1:k]))  mvps=$(mvps)")
        end
    end

    # Final extraction
    Yk = Y[:, 1:k]
    Vk = V_m * Yk

    V = Vk
    D = Diagonal(theta[1:k])
    # ritz = (theta = theta, res = res)
    info = (it = iters, mvps = mvps, res = res)
    return V, D, ritz, info
end

# ----------------- Arnoldi with sketch orthogonalization -------------------

function arnoldi_sketch(A, v1::AbstractVector, m::Integer, SS, s::Integer, orth_method::AbstractString)
    n = length(v1);
    T = ComplexF64

    V  = zeros(T, n, m)
    SV = zeros(T, s, m)
    H  = zeros(T, m, m)
    mvps = 0

    V[:, 1]  = v1 / norm(v1)
    SV[:, 1] = SS(V[:, 1])
    ns1 = norm(SV[:, 1])
    SV[:, 1] ./= ns1
    V[:, 1]  ./= ns1

    for j in 1:m-1
        w  = applyA(A, V[:, j]); mvps += 1
        sw = SS(w)

        sw, h = orth_coeffs_sketch(view(SV, :, 1:j), sw, orth_method)
        w  .-= view(V, :, 1:j) * h

        H[1:j, j] .= h
        H[j+1, j] = norm(sw)
        if H[j+1, j] == 0
            V  = V[:, 1:j]; SV = SV[:, 1:j]; H = H[1:j, 1:j]
            return V, SV, H, zero(T), zeros(T, n), mvps
        end
        V[:, j+1]  .= w  / H[j+1, j]
        SV[:, j+1] .= sw / H[j+1, j]
    end

    w  = applyA(A, V[:, m]); mvps += 1
    sw = SS(w)
    sw, h = orth_coeffs_sketch(view(SV, :, 1:m), sw, orth_method)
    H[1:m, m] .= h
    w .-= view(V, :, 1:m) * h

    # Re-sketch V for stability
    SV = SS(V)
    beta = norm(sw)
    f = w
    return V, SV, H, beta, f, mvps
end

function arnoldi_expand_from_block_sketch(A, V::AbstractMatrix, SV::AbstractMatrix,
                                          H_k::AbstractMatrix, m::Integer, SS, s::Integer,
                                          orth_method::AbstractString)
    n, k = size(V)
    T = ComplexF64

    Vexp  = zeros(T, n, m);   Vexp[:, 1:k]  .= V
    SVexp = zeros(T, s, m);   SVexp[:, 1:k] .= SV
    H     = zeros(T, m, m);   H[1:k, 1:k]   .= H_k
    mvps = 0

    for j in k:(m-1)
        w  = applyA(A, Vexp[:, j]); mvps += 1
        sw = SS(w)

        sw, h = orth_coeffs_sketch(view(SVexp, :, 1:j), sw, orth_method)
        w  .-= view(Vexp, :, 1:j) * h

        H[1:j, j] .= h
        H[j+1, j]  = norm(sw)
        if H[j+1, j] == 0
            Vexp  = Vexp[:, 1:j]; SVexp = SVexp[:, 1:j]; H = H[1:j, 1:j]
            return Vexp, SVexp, H, zero(T), zeros(T, n), mvps
        end
        Vexp[:, j+1]  .= w  / H[j+1, j]
        SVexp[:, j+1] .= sw / H[j+1, j]
    end

    w  = applyA(A, Vexp[:, m]); mvps += 1
    sw = SS(w)
    sw, h = orth_coeffs_sketch(view(SVexp, :, 1:m), sw, orth_method)
    H[1:m, m] .= h
    w .-= view(Vexp, :, 1:m) * h

    # Re-sketch V for stability
    SVexp = SS(Vexp)
    beta = norm(sw)
    f = w
    return Vexp, SVexp, H, beta, f, mvps
end

# ----------------- Sketch-only orthogonalization (in sketch space) ---------

function orth_coeffs_sketch(SV::AbstractMatrix, sw::AbstractVector, orth_method::AbstractString)
    j = size(SV, 2)
    j == 0 && return sw, zeros(eltype(SV), 0)

    meth = lowercase(orth_method)
    if meth == "rcgs"
        h  = SV' * sw
        sw -= SV * h
        return sw, h

    elseif meth == "rcgs2"
        h  = SV' * sw
        sw -= SV * h
        h2 = SV' * sw
        sw -= SV * h2
        h += h2
        return sw, h

    elseif meth == "rgs"
        # LS in sketch space (like MATLAB SV\sw)
        h  = SV \ sw
        sw -= SV * h
        return sw, h

    else
        error("Unknown orth_method \"$orth_method\". Use rgs, rcgs, or rcgs2.")
    end
end


# ----------------------------- Utilities -----------------------------------

applyA(A::AbstractMatrix, x::AbstractVector) = A * x
applyA(A::Function,      x::AbstractVector) = A(x)

function infer_n_and_v0(A, v0)
    if v0 !== nothing
        v = vec(v0); n = length(v)
    else
        if A isa AbstractMatrix
            n = size(A, 1)
            v = randn(n)
        else
            error("For function-handle A, please pass v0 to infer n.")
        end
    end
    v ./= max(norm(v), eps(real(eltype(v))))
    return n, v
end

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