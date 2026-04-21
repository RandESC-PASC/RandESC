using Test
using RandESC
using Random
using LinearAlgebra



function testMatrix(A, n, k, testName; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    # compute true results
    ea = eigen(A)
    evals = ea.values
    evecs = ea.vectors
    evals = evals[1:k]
    evecs = evecs[:, 1:k]
    v0 = randn(eltype(A), n, k)

    # testing block randomized JD
    @testset "$testName: testing jd_sketched" begin
        V_rjdb, lambda_rjdb, history_rjdb = jd_sketched(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        rjdb_passed = maximum(abs.(evals .- lambda_rjdb)) < test_tol
        if !rjdb_passed
            @warn "$testName: jd_sketched eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rjdb)))"
        end
        @test rjdb_passed
        gram_rjdb = V_rjdb' * V_rjdb
        rjdb_orth_passed = maximum(abs.(gram_rjdb - I)) < test_tol
        if !rjdb_orth_passed
            @warn "$testName: jd_sketched eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rjdb - I)))"
        end
        @test rjdb_orth_passed
    end

    # testing standard block JD
    @testset "$testName: testing jd" begin
        V_jdb, lambda_jdb, _ = jd(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        jdb_passed = maximum(abs.(evals .- lambda_jdb)) < test_tol
        if !jdb_passed
            @warn "$testName: jd eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_jdb)))"
        end
        @test jdb_passed
        gram_jdb = V_jdb' * V_jdb
        jdb_orth_passed = maximum(abs.(gram_jdb - I)) < test_tol
        if !jdb_orth_passed
            @warn "$testName: jd eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_jdb - I)))"
        end
        @test jdb_orth_passed
    end
end

@testset "RandESC Symmetric Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    # small test with random spd matrix
    ntests = 7
    test_tol = 1e-5
    iter_tol = 1e-8

    for test in 1:ntests
        n = (test + 3) * (test + 3)
        k = max(1, ceil(Int, 0.05 * n))
        A = randn(Float64, n, n)
        A = A + A'
        testMatrix(A, n, k, "Small random spd matrix test $test", test_tol=test_tol, iter_tol=iter_tol, verbose=false)
    end
end


@testset "RandESC Hermitian Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    # small test with random spd matrix
    ntests = 7
    test_tol = 1e-5
    iter_tol = 1e-8

    for test in 1:ntests
        n = (test + 3) * (test + 3)
        k = max(1, ceil(Int, 0.05 * n))
        A = randn(ComplexF64, n, n)
        A = A + A'
        testMatrix(A, n, k, "Small random spd matrix test $test", test_tol=test_tol, iter_tol=iter_tol, verbose=false)
    end
end

# Tolerances are pretty loose for single precision tests, otherwise they fail
@testset "RandESC Symmetric Single Precision Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    ntests = 8
    test_tol = 1e-3
    iter_tol = 1e-3

    for test in 1:ntests
        n = (test + 3) * (test + 3)
        k = max(1, ceil(Int, 0.05 * n))
        A = randn(Float32, n, n)
        A = A + A'
        # iter_tol is multiplied by n here since the norm of the residual is extensive.
        testMatrix(A, n, k, "Small random symmetric Float32 matrix test $test", test_tol=test_tol * n, iter_tol=iter_tol, verbose=false)
    end
end

# Tolerances are pretty loose for single precision tests, otherwise they fail
@testset "RandESC Hermitian Single Precision Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    ntests = 8
    test_tol = 1e-3
    iter_tol = 1e-3

    for test in 1:ntests
        n = (test + 3) * (test + 3)
        k = max(1, ceil(Int, 0.05 * n))
        A = randn(ComplexF32, n, n)
        A = A + A'
        # iter_tol is multiplied by n here since the norm of the residual is extensive.
        testMatrix(A, n, k, "Small random Hermitian ComplexF32 matrix test $test", test_tol=test_tol, iter_tol=iter_tol * n, verbose=false)
    end
end