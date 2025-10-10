include("sketch.jl")
include("IRA.jl")
include("RIRA.jl")
include("LOBPCG.jl")

using Random
Random.seed!(98423598)

n = 50
k = 7

ntests = 10
test_tol = 1e-5
iter_tol = 1e-8

for test in 1:ntests
    println("test $test")
    # example spd matrix
    A = randn(n, n)
    A = A' * A

    # compute true results
    evals = eigvals(A)
    evals = evals[1:k]
    evecs = eigvecs(A)
    evecs = evecs[:, 1:k]

    V_ira, D_ira, ritz_ira, info_ira = ira(A, n, k; verbose=false, tol=iter_tol)
    lambda_ira = diag(D_ira)
    V_rira, D_rira, ritz_rira, info_rira = rand_ira(A, k; verbose=false, tol=iter_tol)
    lambda_rira = diag(D_rira)
    V_lobpcg, Lambda_lobpcg, info_lobpcg = lobpcg(A, n, k; verbosity=0, tol=iter_tol)
    lambda_lobpcg = Lambda_lobpcg

    # check results
    for i in 1:k
        err_ira = minimum(abs.(evals .- lambda_ira[i])) / abs(evals[i])
        err_rira = minimum(abs.(evals .- lambda_rira[i])) / abs(evals[i])
        err_lobpcg = minimum(abs.(evals .- lambda_lobpcg[i])) / abs(evals[i])
        if err_ira > test_tol
            error("rand_ira failed with rel err $err_ira for eigenvalue $i")
        end
        if err_rira > test_tol
            error("rand_rira failed with rel err $err_rira for eigenvalue $i")
        end
        if err_lobpcg > test_tol
            error("lobpcg failed with rel err $err_lobpcg for eigenvalue $i")
        end
    end
end
println("all $ntests tests passed")