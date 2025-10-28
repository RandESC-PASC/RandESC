# LOBPCG_sketch.jl
# randomized LOBPCG with sketched orthogonalization + soft locking
# expects `sketch(n, s, sketch_type; seed)` to exist and return a callable SS: X -> S*X
# (defined in your sketch.jl)

using LinearAlgebra, Random

# ------------------------------------------------------------
# top-level solver
# ------------------------------------------------------------
function lobpcg_sketch_lock(A, n::Integer, k::Integer;
                            M=nothing, X0=nothing, tol::Real=1e-8, maxit::Integer=100,
                            verbosity::Integer=1, ritz_order::AbstractString="smallest",
                            proj_method::AbstractString="rgs", normA=nothing,
                            precond_preparation=nothing, lock_tol::Real=tol,
                            sketch_type::AbstractString="sparsestack",
                            sketch_s::Union{Nothing,Int}=nothing,
                            sketch_seed::Union{Nothing,Int}=nothing)

    n = Int(n); k = Int(k)
    @assert 1 ≤ k ≤ n "k must be between 1 and n"

    # ---- sketch operator (callable): SS(X) = S * X ----
    s  = isnothing(sketch_s) ? min(n, 6k) : Int(sketch_s)
    SS = sketch(n, s, sketch_type; seed=sketch_seed)  # <— your factory should return a callable

    # ---- init X and sketched-orthonormalize ----
    X = X0 === nothing ? randn(n, k) : copy(X0)
    X = sketch_orth_onlyV(X, SS, s, proj_method)      # returns only V

    # ---- rough ||A|| for relative residuals bookkeeping ----
    mvps = 0
    nA = isnothing(normA) ? begin
        tmp = randn(n, 5) |> complex
        v = A * tmp; mvps += 5
        norm(v) / norm(tmp)
    end : normA

    AX = similar(X); mul!(AX, A, X); mvps += size(X,2)

    # ---- initial Rayleigh–Ritz in span(X) ----
    RfactC = cholesky(X'*X)
    Ahat = RfactC.L \ (Hermitian(X'*AX) / RfactC.U)
    C, Λ = rayleigh_ritz_simple(Ahat; pickSmallest = lowercase(ritz_order) == "smallest")
    X = X * (UpperTriangular(RfactC.U) \ C[:, 1:k])
    AX = AX * (UpperTriangular(RfactC.U) \ C[:, 1:k])

    Lambda = Λ[1:k]
    R = AX - X * Diagonal(Lambda)

    lock_smallest = lowercase(ritz_order) == "smallest"
    lock_front = lock_smallest ? 0      : k + 1   # if "largest", move lock_front downward
    rnorm  = zeros(Float64, k)

    # ---- search directions ----
    T  = eltype(X)
    P  = Array{T}(undef, n, 0)
    AP = Array{T}(undef, n, 0)

    # ---- AW reuse buffer ----
    AWbuf = Array{T}(undef, n, k)

    it = 0
    while it < maxit
        it += 1

        # residual norms & (re)lock
        @inbounds @views for j in 1:k
            rnorm[j] = norm(R[:, j]) #/ (nA+abs(Lambda[j]))
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

        # precondition residuals (active only)
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

        # randomized/sketched projection of W against span(X) and span(P)
        rgs_project_out_multi!(W, SS, s, X, P; method = proj_method, final_orth = proj_method)

        # trial subspace Sfull = [X, P, W]
        Sfull = hcat(X, P, W)
        @views begin
            AW = AWbuf[:, 1:size(W,2)]
            mul!(AW, A, W); mvps += size(W,2)
        end
        ASfull = hcat(AX, AP, AW)

        RfactC = cholesky(Hermitian(Sfull'*Sfull))
        Ahat = RfactC.L \ (Hermitian(Sfull'*ASfull) / RfactC.U)
        C, Λ = rayleigh_ritz_simple(Ahat; pickSmallest = lowercase(ritz_order) == "smallest")

        

        # keep k best
        mul!(X,  Sfull, RfactC.U \ C[:, 1:k])
        mul!(AX, ASfull, RfactC.U \ C[:, 1:k])
        Lambda = Λ[1:k]

        # residuals
        R = AX - X * Diagonal(Lambda)

        # update search directions for ACTIVE columns only
        sX = k
        sP = size(P, 1) == 0 ? 0 : size(P,2)
        sW = size(W, 1) == 0 ? 0 : size(W,2)

        if sP + sW == 0 || isempty(act)
            P  = Array{T}(undef, n, 0)
            AP = Array{T}(undef, n, 0)
        else
            # Orthogonal combination of [P W] columns corresponding to active Ritz vectors
            @views AblkT = transpose(C[act, sX+1:sX+sP+sW])
            qrf = qr(AblkT)
            t   = min(size(AblkT)...)
            QQ  = Matrix(qrf.Q)[:, 1:t]
            @views CQ  = C[:, sX+1:sX+sP+sW] * QQ
            P  = Sfull  * (UpperTriangular(RfactC.U) \ CQ)
            AP = ASfull * (UpperTriangular(RfactC.U) \ CQ)

            # keep P ~ orthogonal to X in the sketched sense
            # rgs_project_out_multi!(P, SS, s, X; method = proj_method, final_orth = proj_method)
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

# ------------------------------------------------------------
# randomized/sketched helpers
# ------------------------------------------------------------

# convenience: return only the orthonormalized V (not SV)
sketch_orth_onlyV(V::AbstractMatrix, S::Function, s::Integer, method::AbstractString="rcgs2") =
    first(sketch_orth!(copy(V), S, s, method))

# sketched orthogonalization (mutating returns (Vt, SVt); V is not modified)
function sketch_orth!(V::AbstractMatrix, S::Function, s::Integer, orth_method::AbstractString="rcgs2")
    n, j = size(V)
    if j == 0
        return Matrix{eltype(V)}(undef, n, 0), Matrix{Float64}(undef, s, 0)
    end

    meth = lowercase(orth_method)
    @assert meth in ("rgs","rcgs","rcgs2") "Unknown orth_method \"$orth_method\""

    # infer sketch eltype from first column
    s1 = S(@view V[:,1]); @assert length(s1) == s "Sketch size mismatch"
    Ts = eltype(s1)

    Vt  = Matrix{eltype(V)}(undef, n, j)
    SVt = Matrix{Ts}(undef, s, j)

    # first column
    n1 = norm(s1); @assert n1 > 0 "First column has zero sketch norm."
    @views Vt[:,1]  .= V[:,1] ./ n1
    @views SVt[:,1] .= s1     ./ n1

    # remaining columns
    for i in 2:j
        v = copy(@view V[:, i])
        @views Vprev  = Vt[:, 1:i-1]
        @views SVprev = SVt[:, 1:i-1]

        if meth == "rgs"
            h = SVprev \ S(v)
            v .-= Vprev * h
        else
            svec = S(v)
            h = SVprev' * svec
            v .-= Vprev * h
            if meth == "rcgs2"
                svec = S(v)
                h2 = SVprev' * svec
                v .-= Vprev * h2
            end
        end

        svec = S(v)
        ni = norm(svec); @assert ni > 0 "Dependent column encountered after projection."
        @views Vt[:, i]  .= v    ./ ni
        @views SVt[:, i]  .= svec ./ ni
    end

    return Vt, SVt
end

# randomized/sketched block projection (in-place on Z), final sketched orthonormalization
function rgs_project_out_multi!(Z::AbstractMatrix, S::Function, s::Integer, Us::AbstractMatrix...;
                                method::AbstractString = "rgs",
                                final_orth::AbstractString = "rcgs2")
    q = size(Z,2)
    q == 0 && return Z

    m = lowercase(method)
    @assert m in ("rgs","rcgs","rcgs2") "Unknown proj method \"$method\" (use rgs, rcgs, rcgs2)"

    SZ = S(Z)
    for U in Us
        p = size(U,2); p == 0 && continue
        SU = S(U)
        if m == "rgs"
            H = SU \ SZ
            Z .-= U * H
            SZ  = S(Z)
        elseif m == "rcgs"
            H = adjoint(SU) * SZ
            Z .-= U * H
            SZ  = S(Z)
        else
            H1 = adjoint(SU) * SZ
            Z  .-= U * H1
            SZ  =  S(Z)
            H2 = adjoint(SU) * SZ
            Z  .-= U * H2
            SZ  =  S(Z)
        end
    end

    Vt, _ = sketch_orth!(Z, S, s, final_orth)
    copy!(Z, Vt)
    return Z
end

function rayleigh_ritz_mod(Ahat::AbstractMatrix, C::Cholesky; pickSmallest::Bool=true)
    G = C.L\(Ahat/C.U)
    G = (G+G')/2
    F = eigen(Hermitian(G))
    d = real(F.values)
    V = F.vectors
    idx = pickSmallest ? sortperm(d) : sortperm(d, rev=true)
    return V[:, idx], d[idx]
end

function rayleigh_ritz_simple(Ahat::AbstractMatrix; pickSmallest::Bool=true)
    F = eigen(Hermitian(Ahat))
    d = real(F.values)
    V = F.vectors
    idx = pickSmallest ? sortperm(d) : sortperm(d, rev=true)
    return V[:, idx], d[idx]
end


# symmetric-definite Rayleigh–Ritz in span(X)
function rayleigh_ritz(X::AbstractMatrix, AX::AbstractMatrix; pickSmallest::Bool=true)
    XAX = X' * AX
    Ahat = (XAX + XAX') / 2
    XX   = X' * X
    # F = eigen(Hermitian(Ahat), Hermitian(XX))
    F = eigen(Hermitian(Ahat))
    d = real(F.values)
    V = F.vectors
    idx = pickSmallest ? sortperm(d) : sortperm(d, rev=true)
    # println([cond(V),norm(V'*V-I),norm(V'*XX*V-I)])
    return V[:, idx], d[idx]
end

function sketch_rayleigh_ritz(X::AbstractMatrix, AX::AbstractMatrix, SS::Function; pickSmallest::Bool=true)
    Ahat = SS(X)' * SS(AX)
    # F = eigen(Hermitian(Ahat), Hermitian(XX))
    F = eigen(Ahat)
    d = real(F.values)
    V = F.vectors
    idx = pickSmallest ? sortperm(d) : sortperm(d, rev=true)
    # println([cond(V),norm(V'*V-I)])
    return V[:, idx], d[idx]
end

