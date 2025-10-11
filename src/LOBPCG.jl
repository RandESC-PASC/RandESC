# lobpcg.jl
using LinearAlgebra, Random

"""
    lobpcg(A, n, k; M=nothing, X0=nothing, tol=1e-6, maxit=200,
                 verbosity=0, ritz_order="smallest", proj_method="CGS2")

Simplified LOBPCG for the standard symmetric/Hermitian eigenproblem
    A x = λ x
(no locking/deflation).

Arguments
- `A`  :: AbstractMatrix or function `X -> A*X` that applies A to a dense block X.
- `n`  :: Problem size (used if `X0` is not given and `A` is a function).
- `k`  :: Number of desired eigenpairs.

Keywords (all optional)
- `M`           :: preconditioner matrix or function `R -> approx(M\\R)` (default `nothing` ⇒ identity)
- `X0`          :: `n×k` initial guess (default random)
- `tol`         :: relative residual tolerance (default `1e-6`)
- `maxit`       :: maximum iterations (default `200`)
- `verbosity`   :: 1=per-iter summary (default), 0=silent
- `ritz_order`  :: `"smallest"` (default) or `"largest"`
- `proj_method` :: `"CGS" | "CGS2"` (default) | `"MGS" | "MGS2"`
- `normA`      :: estimate of ||A||_2 (for relative residuals; default `nothing` ⇒ computed internally)
- `precond_preparation` :: optional function that has to be called everytime before the preconditioner M is applied (e.g. to update internal data structures).
                             it takes the preconditioner M and the current Ritz vectors X as arguments.

Returns
- `X`       :: `n×k` matrix of orthonormal Ritz vectors
- `Lambda`  :: `k`-vector of Ritz values (ascending if `ritz_order="smallest"`)
- `info`    :: NamedTuple: `(it, relres, mvps)`
              where `relres` is an `it×k` matrix of per-vector relative residuals.
"""
function lobpcg(A, n::Integer, k::Integer;
                M=nothing, X0=nothing, tol::Real=1e-8, maxit::Integer=200,
                verbosity::Integer=1, ritz_order::AbstractString="smallest",
                proj_method::AbstractString="CGS2", normA = nothing, precond_preparation = nothing)

    if isnothing(M)
        M = I
    end

    # ---- initial guess X (n×k) and orthonormalize ----
    X = X0 === nothing ? randn(n, k) : copy(X0)
    X = qr_orthonormalize(X)

    # ---- rough norm(A) estimate (Frobenius) for relres scaling ----
    if isnothing(normA)
        tempX = randn(n, 5) 
        nA = norm(A * tempX)   # Frobenius/Frobenius
        nA = nA / norm(tempX)  # estimate of ||A||_2
        mvps = 5
    else
        nA = normA
        mvps = 0
    end

    AX = similar(X)
    # ---- initial Rayleigh–Ritz in span(X) ----
    mul!(AX, A, X); mvps += size(X, 2)
    C, Lambda = rayleigh_ritz_standard(X, AX; pickSmallest = lowercase(ritz_order) == "smallest")
    X  = X * C[:, 1:k]
    AX = AX * C[:, 1:k]
    Lambda = Lambda[1:k]
    R = AX - X * Diagonal(Lambda)

    # residual history
    # relres = zeros(0, k)
    rel = zeros(k,1);

    # no locking/deflation; search directions start empty
    T = eltype(X)
    P  = Array{T}(undef, size(X,1), 0)
    AP = Array{T}(undef, size(X,1), 0)

    AW = similar(X)
    W = similar(X)

    it = 0
    while it < maxit
        it += 1

        # ---- precondition & project out (in the full space) ----
        mul!(W, M, R)
        W  = gs_project_out(W, hcat(X, P), proj_method)   # returns orthonormalized W

        # ---- trial subspace and its image under A ----
        S  = hcat(X, P, W)
        # AW = Afun(W); mvps += size(W, 2)
        mul!(AW, A, W); mvps += size(W, 2)
        AS = hcat(AX, AP, AW)

        # ---- Rayleigh–Ritz on S ----
        C, Λ = rayleigh_ritz_standard(S, AS; pickSmallest = lowercase(ritz_order) == "smallest")
        # X   = S  * C[:, 1:k]
        mul!(X, S, C[:, 1:k])
        # AX  = AS * C[:, 1:k]
        mul!(AX, AS, C[:, 1:k])
        Lambda = Λ[1:k]

        # ---- residuals R = AX - X*Λ, and relative residuals per vector ----
        R = AX - X * Diagonal(Lambda)
        rnorm = [norm(view(R, :, j)) for j in 1:k]
        axnorm = nA .+ abs.(Lambda)           # simple safe scaling (like MATLAB line)
        rel = rnorm ./ axnorm
        # relres = vcat(relres, permutedims(rel))  # append row

        if maximum(rel) <= tol
            break
        end

        # ---- update search directions P from C(1:k, k+1:end) via thin QR ----
        # C is size s×s, with s = size(S,2). Let Ablk' = C(1:k, k+1:end)' ( (s-k)×k )
        AblkT = transpose(C[1:k, k+1:end])
        qrf   = qr(AblkT)
        t     = min(size(AblkT)...)               # economy columns
        QQ    = Matrix(qrf.Q)[:, 1:t]
        CQ    = C[:, k+1:end] * QQ                # s×t

        P  = S  * CQ
        AP = AS * CQ

        if verbosity > 0
            println("LOBPCG iter ", it, ": max relres = ", maximum(rel), ", mvps = ", mvps)
        end
    end

    info = (it = it, mvps = mvps, res = rel)
    return X, Lambda, info
end

# ============================ helpers ======================================

# Economy QR: return orthonormal columns spanning cols(V)
function qr_orthonormalize(V::AbstractMatrix)
    size(V, 2) == 0 && return V
    F = qr(V)                         # Q is m×m; take first k columns
    k = size(V, 2)
    Q = Matrix(F.Q)[:, 1:k]
    return Q
end

# Project Z out of span(U) by GS variant, then orthonormalize Z
function gs_project_out(Z::AbstractMatrix, U::AbstractMatrix, method::AbstractString)
    if isempty(U) || size(U,2) == 0 || size(Z,2) == 0
        return qr_orthonormalize(Z)
    end
    meth = uppercase(method)
    if meth == "CGS"
        # H = U' * Z;  Z = Z - U * H
        mul!(H, U', Z);  mul!(Z, U, H, -1, 1)
    elseif meth == "CGS2"
        H = U' * Z;  Z = Z - U * H
        H = U' * Z;  Z = Z - U * H
    elseif meth == "MGS"
        for j in 1:size(U,2)
            h = U[:, j]' * Z
            Z .-= U[:, j] * h
        end
    elseif meth == "MGS2"
        for _ in 1:2
            for j in 1:size(U,2)
                h = U[:, j]' * Z
                Z .-= U[:, j] * h
            end
        end
    else
        error("Unknown proj_method: $method")
    end
    return qr_orthonormalize(Z)
end

# Rayleigh–Ritz in span(X): diagonalize Ahat = sym(X'AX), return eigenvectors C and eigenvalues λ
function rayleigh_ritz_standard(X::AbstractMatrix, AX::AbstractMatrix; pickSmallest::Bool=true)
    XAX = X' * AX
    Ahat = (XAX + XAX') / 2                       # symmetrize for safety
    F = eigen(Hermitian(Ahat))                    # guarantees real eigenvalues
    d = real(F.values)
    V = F.vectors
    idx = pickSmallest ? sortperm(d) : sortperm(d, rev=true)
    return V[:, idx], d[idx]
end
