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

    # ---- initial guess X (n×k) and orthonormalize ----
    X = X0 === nothing ? randn(n, k) : copy(X0)
    X = qr_orthonormalize!(X)

    # ---- rough norm(A) estimate (Frobenius) for relres scaling ----
    if isnothing(normA)
        tempX = complex(randn(n, 5) )
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
        if !isnothing(M)
            if !isnothing(precond_preparation)
                precond_preparation(M, X)
            end
            ldiv!(W, M, R)
        else
            copy!(W, R)   # identity preconditioner
        end
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

        if maximum(rnorm) <= tol
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
qr_orthonormalize(V::AbstractMatrix) = qr_orthonormalize!(copy(V))
# Economy QR: return orthonormal columns spanning cols(V)
function qr_orthonormalize!(V::AbstractMatrix)
    m, n = size(V)
    n == 0 && return similar(V, eltype(V), m, 0)
    A = V isa StridedMatrix ? V : Matrix(V)
    F = qr!(A, Val(true))

    return Matrix(F.Q)
end

# Project Z out of span(U) by GS variant, then orthonormalize Z
function gs_project_out!(Z::AbstractMatrix, U::AbstractMatrix;
                         method::AbstractString = "CGS2")

    # break down
    if size(Z,2) == 0 || size(U,2) == 0
        return qr_orthonormalize!(Z)
    end
    @assert size(U,1) == size(Z,1) "U and Z must have the same number of rows"

    Zs = Z isa StridedMatrix ? Z : Matrix(Z)
    Us = U isa StridedMatrix ? U : Matrix(U)

    r, c = size(Us,2), size(Zs,2)
    H =  Matrix{eltype(Zs)}(undef, r, c)

    # Choose algorithm
    m = lowercase(method)
    if m == "cgs"
        mul!(H, adjoint(Us), Zs)            # H = U'Z
        mul!(Zs, Us, H, -one(eltype(Zs)),  one(eltype(Zs)))  # Z -= U*H
    elseif m == "cgs2"
        mul!(H, adjoint(Us), Zs)            # 1st pass
        mul!(Zs, Us, H, -one(eltype(Zs)),  one(eltype(Zs)))
        mul!(H, adjoint(Us), Zs)            # 2nd pass
        mul!(Zs, Us, H, -one(eltype(Zs)),  one(eltype(Zs)))
    elseif m == "mgs" || m == "mgs2"
        passes = m == "mgs2" ? 2 : 1
        for _ in 1:passes
            @inbounds for j in 1:size(Us,2)
                u = view(Us, :, j)
                hv = (u' * Zs)
                mul!(Zs, u, hv, -one(eltype(Zs)), one(eltype(Zs)))
            end
        end
    else
        error("Unknown proj_method: $method (use \"CGS\", \"CGS2\", \"MGS\", or \"MGS2\")")
    end

    return qr_orthonormalize!(Zs)
end

gs_project_out(Z::AbstractMatrix, U::AbstractMatrix, method::AbstractString) =
    gs_project_out!(copy(Z), U; method)

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

function lobpcg_softlock(A, n::Integer, k::Integer;
                         M=nothing, X0=nothing, tol::Real=1e-8, maxit::Integer=200,
                         verbosity::Integer=1, ritz_order::AbstractString="smallest",
                         proj_method::AbstractString="CGS2", normA = nothing,
                         precond_preparation = nothing, lock_tol::Real=tol)

    # -- initial guess X (n×k) and orthonormalize --
    X = X0 === nothing ? randn(n, k) : copy(X0)
    X = qr_orthonormalize!(X)

    # -- rough ||A|| estimate for relative residuals --
    if isnothing(normA)
        tempX = complex(randn(n, 5))
        nA = norm(A * tempX); nA /= norm(tempX)
        mvps = 5
    else
        nA = normA
        mvps = 0
    end

    AX = similar(X)
    mul!(AX, A, X); mvps += size(X,2)

    # initial RR in span(X)
    C, Lambda = rayleigh_ritz_standard(X, AX; pickSmallest = lowercase(ritz_order) == "smallest")
    X  = X * C[:, 1:k]
    AX = AX * C[:, 1:k]
    Lambda = Lambda[1:k]
    R = AX - X * Diagonal(Lambda)

    # soft-lock bookkeeping
    locked = falses(k)                
    rel   = zeros(Float64, k)    
    rnorm = zeros(Float64, k)     

    # search directions (active-only)
    T  = eltype(X)
    P  = Array{T}(undef, size(X,1), 0)
    AP = Array{T}(undef, size(X,1), 0)

    # reusable buffers to reduce allocs
    AWbuf = Array{T}(undef, size(X,1), k)   # max width k; we’ll take views as needed 

    it = 0
    while it < maxit
        it += 1

        # ---- residuals & (re)lock ----
        @inbounds @views for j in 1:k               
            rnorm[j] = norm(R[:, j])
        end
        axnorm = nA .+ abs.(Lambda)                 # small alloc; could be reused if wanted
        rel .= rnorm #./ axnorm

        locked .= rnorm .<= lock_tol

        if all(locked)                              
            if verbosity > 0
                println("LOBPCG iter $it: all $k columns locked; mvps = $mvps")
            end
            break
        end

        # ---- indices of active (unlocked) columns ----
        act = findall(!, locked)                    

        # ---- precondition residuals of active columns only ----
        @views Ract = R[:, act]                     # active columns
        if !isnothing(M)
            if !isnothing(precond_preparation)
                precond_preparation(M, X)
            end
            W = similar(Ract)
            ldiv!(W, M, Ract)                       
        else
            W = copy(Ract)                          
        end

        # ---- project W against span(X) and span(P) ----
        W = gs_project_out_multi(W, X, P; method = proj_method)

        # ---- trial subspace S = [X, P, W] (locked vectors stay inside X) ----
        S  = hcat(X, P, W)
        @views begin
            AW = AWbuf[:, 1:size(W,2)]              # reuse AW buffer (preallocation)
            mul!(AW, A, W); mvps += size(W, 2)      # matvecs only for active block
        end
        AS = hcat(AX, AP, AW)

        # ---- Rayleigh–Ritz on S ----
        C, Λ = rayleigh_ritz_standard(S, AS; pickSmallest = lowercase(ritz_order) == "smallest")

        # update Ritz vectors/values
        mul!(X,  S,  C[:, 1:k])
        mul!(AX, AS, C[:, 1:k])
        Lambda = Λ[1:k]

        # recompute residuals for all k kept Ritz pairs (locked included)
        R = AX - X * Diagonal(Lambda)

        # ---- update search directions P for ACTIVE Ritz vectors only ----
        sX = k
        sP = size(P,2)
        sW = size(W,2)

        if sP + sW == 0
            P  = Array{T}(undef, size(X,1), 0)
            AP = Array{T}(undef, size(X,1), 0)
        else
            if !isempty(act)
                #  use only the rows of the X-block that correspond to ACTIVE columns
                @views AblkT = transpose(C[act, sX+1:sX+sP+sW])
                qrf = qr(AblkT)
                t   = min(size(AblkT)...)
                QQ  = Matrix(qrf.Q)[:, 1:t]
                @views CQ  = C[:, sX+1:sX+sP+sW] * QQ

                P  = S  * CQ
                AP = AS * CQ
            else
                P  = Array{T}(undef, size(X,1), 0)
                AP = Array{T}(undef, size(X,1), 0)
            end
        end

        if verbosity > 0
            maxrel_active = isempty(act) ? 0.0 : maximum(rel[act]) 
            println("LOBPCG (soft) iter ", it,
                    ": max relres(active) = ", maxrel_active,
                    ", locked = ", count(locked), "/", k,
                    ", mvps = ", mvps)
        end
    end

    if verbosity > 0
        println("LOBPCG finished in $it iterations, locked = ", count(locked), "/$k, total mvps = $mvps")
        # println("Maximum final relres: ", maximum(rel), " tolerance was $tol")
    end
    if tol < maximum(rel)
        @warn "Warning: Not all requested eigenpairs converged to the desired tolerance $tol, maximum relres: $(maximum(rel)). tolerance was $tol"
    end

    info = (it = it, mvps = mvps, res = rel, locked = locked) 
    return X, Lambda, info
end

# for orthogonalizing against multiple blocks instead of concatenating and allocating memory
function gs_project_out_multi!(Z::AbstractMatrix, Us::AbstractMatrix...;
                               method::AbstractString = "CGS2")
    # trivial cases
    if size(Z,2) == 0
        return qr_orthonormalize!(Z)
    end
    Zs = Z isa StridedMatrix ? Z : Matrix(Z)

    m = lowercase(method)
    if m == "cgs"
        for U in Us
            if size(U,2) == 0; continue; end
            Us_ = U isa StridedMatrix ? U : Matrix(U)
            H = Matrix{eltype(Zs)}(undef, size(Us_,2), size(Zs,2))
            mul!(H, adjoint(Us_), Zs)                          # H = U'Z
            mul!(Zs, Us_, H, -one(eltype(Zs)), one(eltype(Zs)))# Z -= U*H
        end
        return qr_orthonormalize!(Zs)
    elseif m == "cgs2"
        # two passes for each U (CGS2), single QR at end
        for U in Us
            if size(U,2) == 0; continue; end
            Us_ = U isa StridedMatrix ? U : Matrix(U)
            H = Matrix{eltype(Zs)}(undef, size(Us_,2), size(Zs,2))
            # 1st pass
            mul!(H, adjoint(Us_), Zs)
            mul!(Zs, Us_, H, -one(eltype(Zs)), one(eltype(Zs)))
            # 2nd pass
            mul!(H, adjoint(Us_), Zs)
            mul!(Zs, Us_, H, -one(eltype(Zs)), one(eltype(Zs)))
        end
        return qr_orthonormalize!(Zs)
    elseif m == "mgs" || m == "mgs2"
        passes = m == "mgs2" ? 2 : 1
        for _ in 1:passes
            for U in Us
                if size(U,2) == 0; continue; end
                Us_ = U isa StridedMatrix ? U : Matrix(U)
                @inbounds for j in 1:size(Us_,2)
                    u = view(Us_, :, j)
                    hv = (u' * Zs)
                    mul!(Zs, u, hv, -one(eltype(Zs)), one(eltype(Zs)))
                end
            end
        end
        return qr_orthonormalize!(Zs)
    else
        error("Unknown proj_method: $method (use \"CGS\", \"CGS2\", \"MGS\", or \"MGS2\")")
    end
end

# non-mutating wrapper
gs_project_out_multi(Z::AbstractMatrix, Us::AbstractMatrix...; method::AbstractString="CGS2") =
    gs_project_out_multi!(copy(Z), Us...; method)


function lobpcg_lock(A, n::Integer, k::Integer;
                     M=nothing, X0=nothing, tol::Real=1e-8, maxit::Integer=200,
                     verbosity::Integer=1, ritz_order::AbstractString="smallest",
                     proj_method::AbstractString="CGS2", normA = nothing,
                     precond_preparation = nothing)

    # --- initial guess & orthonormalize ---
    X = X0 === nothing ? randn(n, k) : copy(X0)
    X = qr_orthonormalize!(X)

    # --- ||A|| estimate for (optional) diagnostics, matvec counter ---
    if isnothing(normA)
        tmp = complex(randn(n, 5))
        nA  = norm(A * tmp) / norm(tmp)
        mvps = 5
    else
        nA  = normA
        mvps = 0
    end

    # --- initial RR in span(X) ---
    AX = similar(X)
    mul!(AX, A, X); mvps += size(X,2)
    C, Λ = rayleigh_ritz_standard(X, AX; pickSmallest = lowercase(ritz_order) == "smallest")
    X  = X * C[:, 1:k]
    AX = AX * C[:, 1:k]
    Lambda = Λ[1:k]
    R = AX - X * Diagonal(Lambda)

    # --- workspaces & search dirs ---
    T = eltype(X)
    rnorm = zeros(Float64, k)             # absolute residuals per column
    P  = Array{T}(undef, n, 0)
    AP = Array{T}(undef, n, 0)
    AWbuf = Array{T}(undef, n, k)         # reuse for A*W

    # --- ordered locking frontier ---
    lock_smallest = lowercase(ritz_order) == "smallest"
    lock_front = lock_smallest ? 0      : k + 1   # if "largest", move lock_front downward

    it = 0
    while it < maxit
        it += 1

        # ---- residuals (absolute) ----
        @inbounds @views for j in 1:k
            rnorm[j] = norm(R[:, j])
        end

        # ---- advance frontier in-order (monotone) ----
        if lock_smallest
            while lock_front < k && rnorm[lock_front + 1] <= tol
                lock_front += 1
            end
            act = (lock_front + 1):k                # active tail
            done = (lock_front == k)
        else
            while lock_front > 1 && rnorm[lock_front - 1] <= tol
                lock_front -= 1
            end
            act = 1:(lock_front - 1)                # active head (largest-first)
            done = (lock_front == 1)
        end

        if done
            verbosity > 0 && println("LOBPCG (lock) iter $it: all $k columns ≤ tol; mvps = $mvps")
            break
        end

        # ---- precondition residuals of active block only ----
        @views Ract = R[:, act]                     # n × ka (ka may be 0)
        W = similar(Ract)
        if !isnothing(M)
            if !isnothing(precond_preparation)
                precond_preparation(M, X)
            end
            ldiv!(W, M, Ract)
        else
            copy!(W, Ract)
        end

        # ---- project W against span(X,P) (single QR at end) ----
        W = gs_project_out_multi(W, X, P; method = proj_method)

        # ---- trial subspace S = [X, P, W] and AS = [AX, AP, AW] ----
        S  = hcat(X, P, W)
        @views AW = AWbuf[:, 1:size(W,2)]
        mul!(AW, A, W); mvps += size(W,2)
        AS = hcat(AX, AP, AW)

        # ---- Rayleigh–Ritz on S ----
        C, Λ = rayleigh_ritz_standard(S, AS; pickSmallest = lock_smallest)

        # update k Ritz pairs
        mul!(X,  S,  C[:, 1:k])
        mul!(AX, AS, C[:, 1:k])
        Lambda = Λ[1:k]

        # new residuals for next cycle
        R = AX - X * Diagonal(Lambda)

        # ---- update search directions P/AP only for ACTIVE rows of X-block ----
        sX = k
        sP = size(P,2)
        sW = size(W,2)
        if sP + sW == 0
            P  = Array{T}(undef, n, 0)
            AP = Array{T}(undef, n, 0)
        else
            if isempty(act)
                P  = Array{T}(undef, n, 0)
                AP = Array{T}(undef, n, 0)
            else
                @views AblkT = transpose(C[act, sX+1:sX+sP+sW])   # ((sP+sW) × |act|)
                qrf = qr(AblkT)
                t   = min(size(AblkT)...)
                QQ  = Matrix(qrf.Q)[:, 1:t]
                @views CQ = C[:, sX+1:sX+sP+sW] * QQ
                P  = S  * CQ
                AP = AS * CQ
            end
        end

        if verbosity > 0
            max_active = isempty(act) ? 0.0 : maximum(@view rnorm[act])
            println("LOBPCG (lock) iter $it: max absres(active) = $max_active, ",
                    lock_smallest ? "front=" : "back=", lock_front, ", mvps = $mvps")
        end
    end

    if verbosity > 0
        if lock_smallest
            println("LOBPCG (lock) finished in $it iters; locked smallest: $lock_front/$k; mvps=$mvps")
        else
            println("LOBPCG (lock) finished in $it iters; locked largest: $(k-lock_front+1)/$k; mvps=$mvps")
        end
    end

    info = (it = it, mvps = mvps, absres = rnorm, frontier = lock_front)
    return X, Lambda, info
end