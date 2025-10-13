using Test
using RandESC
using Random
using LinearAlgebra
@testset "RandESC Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    n = 50
    k = 7

    ntests = 10
    test_tol = 1e-5
    iter_tol = 1e-8

    for test in 1:ntests
        @testset "Test #$test" begin
            # example spd matrix
            A = randn(n, n)
            A = A' * A

            # compute true results
            ea = eigen(A)
            evals = ea.values
            evecs = ea.vectors
            evals = evals[1:k]
            evecs = evecs[:, 1:k]

            V_ira, D_ira, ritz_ira, info_ira = ira(A, n, k; verbose=false, tol=iter_tol)
            lambda_ira = diag(D_ira)
            V_rira, D_rira, ritz_rira, info_rira = rand_ira(A, n, k; verbose=false, tol=iter_tol)
            lambda_rira = diag(D_rira)
            V_lobpcg, Lambda_lobpcg, info_lobpcg = lobpcg(A, n, k; verbosity=0, tol=iter_tol)
            lambda_lobpcg = Lambda_lobpcg

            # check results
            for i in 1:k
                err_ira = minimum(abs.(evals .- lambda_ira[i])) / abs(evals[i])
                err_rira = minimum(abs.(evals .- lambda_rira[i])) / abs(evals[i])
                err_lobpcg = minimum(abs.(evals .- lambda_lobpcg[i])) / abs(evals[i])
                @test err_ira ≤ test_tol #"IRA failed with rel err $err_ira for eigenvalue $i"
                @test err_rira ≤ test_tol #"rand_IRA failed with rel err $err_rira for eigenvalue $i"
                @test err_lobpcg ≤ test_tol #"lobpcg failed with rel err $err_lobpcg for eigenvalue $i"
            end
        end
    end
end