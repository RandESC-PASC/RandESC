using Test
using RandESC
using Random
using LinearAlgebra



function testMatrix(A, n, k, testName; test_tol=1e-5, iter_tol=1e-8, verbose=false)
    # compute true results
    ea = eigen(A)
    evals = ea.values
    evecs = ea.vectors
    evals = evals[1:k]
    evecs = evecs[:, 1:k]


    V_ira, D_ira, ritz_ira, info_ira = ira(A, n, k; verbose=verbose, tol=iter_tol)
    lambda_ira = diag(D_ira)
    V_rira, D_rira, ritz_rira, info_rira = rand_ira(A, n, k; verbose=verbose, tol=iter_tol)
    lambda_rira = diag(D_rira)
    V_lobpcg, lambda_lobpcg, info_lobpcg = lobpcg(A, n, k; verbosity=verbose ? 1 : 0, tol=iter_tol)
    
    # test eigenvalues
    ira_passed = maximum(abs.(evals .- lambda_ira)) < test_tol
    rira_passed = maximum(abs.(evals .- lambda_rira)) < test_tol
    lobpcg_passed = maximum(abs.(evals .- lambda_lobpcg)) < test_tol

    if !ira_passed
        @warn "$testName: IRA eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_ira)))"
    end
    if !rira_passed
        @warn "$testName: randIRA eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rira)))"
    end
    if !lobpcg_passed
        @warn "$testName: LOBPCG eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_lobpcg)))"
    end
    @testset "$testName: ira eigenvalues" begin
        @test ira_passed
    end
    @testset "$testName: randIRA eigenvalues" begin
        @test rira_passed
    end
    @testset "$testName: LOBPCG eigenvalues" begin
        @test lobpcg_passed 
    end

    # # test eigenvectors 
    gram_ira = V_ira' * V_ira
    gram_rira = V_rira' * V_rira
    gram_lobpcg = V_lobpcg' * V_lobpcg
    ira_orth_passed = maximum(abs.(gram_ira - I)) < test_tol
    rira_orth_passed = maximum(abs.(gram_rira - I)) < test_tol
    lobpcg_orth_passed = maximum(abs.(gram_lobpcg - I)) < test_tol
    if !ira_orth_passed
        @warn "$testName: IRA eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_ira - I)))"
    end
    @testset "$testName: ira eigenvectors orthogonality" begin
        @test ira_orth_passed
    end
    if !rira_orth_passed
        @warn "$testName: randIRA eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rira - I)))"
    end
    @testset "$testName: randIRA eigenvectors orthogonality" begin
        @test rira_orth_passed
    end
    if !lobpcg_orth_passed
        @warn "$testName: LOBPCG eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_lobpcg - I)))"
    end
    @testset "$testName: LOBPCG eigenvectors orthogonality" begin
        @test lobpcg_orth_passed
    end

end

@testset "RandESC Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    n = 50
    k = 7

    ntests = 10
    test_tol = 1e-5
    iter_tol = 1e-8

    for test in 1:ntests
        A = randn(n, n)
        A = A + A'
        testMatrix(A, n, k, "Random symmetric matrix test $test", test_tol=test_tol, iter_tol=iter_tol, verbose=false)
    end
end
