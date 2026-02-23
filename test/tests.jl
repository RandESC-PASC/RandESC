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

    # testing IRA
    @testset "$testName: testing IRA" begin
        V_ira, D_ira, ritz_ira, info_ira = ira(A, n, k; verbose=verbose, tol=iter_tol, maxit=maxiter)
        lambda_ira = diag(D_ira)

        ira_passed = maximum(abs.(evals .- lambda_ira)) < test_tol
        if !ira_passed
            @warn "$testName: IRA eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_ira)))"
        end
        @test ira_passed
        gram_ira = V_ira' * V_ira
        ira_orth_passed = maximum(abs.(gram_ira - I)) < test_tol
        if !ira_orth_passed
            @warn "$testName: IRA eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_ira - I)))"
        end
        @test ira_orth_passed
    end

    # testing randIRA
    @testset "$testName: testing randIRA" begin
        V_rira, D_rira, ritz_rira, info_rira = rand_ira(A, n, k; verbose=verbose, tol=iter_tol, maxit=maxiter)
        lambda_rira = diag(D_rira)
    
        rira_passed = maximum(abs.(evals .- lambda_rira)) < test_tol
        if !rira_passed
            @warn "$testName: randIRA eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rira)))"
        end
        @test rira_passed
        gram_rira = V_rira' * V_rira
        rira_orth_passed = maximum(abs.(gram_rira - I)) < test_tol
        if !rira_orth_passed
            @warn "$testName: randIRA eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rira - I)))"
        end
        @test rira_orth_passed
    end

    # testing LOBPCG
    @testset "$testName: testing LOBPCG" begin
        V_lobpcg, lambda_lobpcg, info_lobpcg = lobpcg(A, n, k; verbosity=verbose ? 1 : 0, tol=iter_tol, maxit=maxiter)
        lobpcg_passed = maximum(abs.(evals .- lambda_lobpcg)) < test_tol
        if !lobpcg_passed
            @warn "$testName: LOBPCG eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_lobpcg)))"
        end
        @test lobpcg_passed
        gram_lobpcg = V_lobpcg' * V_lobpcg
        lobpcg_orth_passed = maximum(abs.(gram_lobpcg - I)) < test_tol
        if !lobpcg_orth_passed
            @warn "$testName: LOBPCG eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_lobpcg - I)))"
        end
        @test lobpcg_orth_passed
    end

    # testing JD
    @testset "$testName: testing JD" begin
        V_jd, lambda_jd, history_jd = jdsym(A; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)
        jd_passed = maximum(abs.(evals .- lambda_jd)) < test_tol
        if !jd_passed
            @warn "$testName: JD eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_jd)))"
        end
        @test jd_passed
        gram_jd = V_jd' * V_jd
        jd_orth_passed = maximum(abs.(gram_jd - I)) < test_tol
        if !jd_orth_passed
            @warn "$testName: JD eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_jd - I)))"
        end
        @test jd_orth_passed
    end

    # testing randomized JD
    @testset "$testName: testing randJD" begin
        V_rjd, lambda_rjd, history_rjd = jdsym_rand(A; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)
        V_rjd, lambda_rjd = sketched_to_fully(A, V_rjd)


        rjd_passed = maximum(abs.(evals .- lambda_rjd)) < test_tol
        if !rjd_passed
            @warn "$testName: randJD eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rjd)))"
        end
        @test rjd_passed
        gram_rjd = V_rjd' * V_rjd
        rjd_orth_passed = maximum(abs.(gram_rjd - I)) < test_tol
        if !rjd_orth_passed
            @warn "$testName: randJD eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rjd - I)))"
        end
        @test rjd_orth_passed
    end

    # testing block randomized JD
    @testset "$testName: testing randJD_block" begin
        V_rjdb, lambda_rjdb, history_rjdb = jdsym_rand_block(A; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        rjdb_passed = maximum(abs.(evals .- lambda_rjdb)) < test_tol
        if !rjdb_passed
            @warn "$testName: randJD_block eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rjdb)))"
        end
        @test rjdb_passed
        gram_rjdb = V_rjdb' * V_rjdb
        rjdb_orth_passed = maximum(abs.(gram_rjdb - I)) < test_tol
        if !rjdb_orth_passed
            @warn "$testName: randJD_block eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rjdb - I)))"
        end
        @test rjdb_orth_passed
    end

    # testing standard block JD (translation of QE cjdsym)
    @testset "$testName: testing jdsym_block" begin
        V_jdb, lambda_jdb, _ = jdsym_block(A; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        jdb_passed = maximum(abs.(evals .- lambda_jdb)) < test_tol
        if !jdb_passed
            @warn "$testName: jdsym_block eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_jdb)))"
        end
        @test jdb_passed
        gram_jdb = V_jdb' * V_jdb
        jdb_orth_passed = maximum(abs.(gram_jdb - I)) < test_tol
        if !jdb_orth_passed
            @warn "$testName: jdsym_block eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_jdb - I)))"
        end
        @test jdb_orth_passed
    end
end

@testset "RandESC Eigenvalue Solver Tests" begin
    Random.seed!(98423598)

    # small test with random spd matrix
    n = 50
    k = 7

    ntests = 10
    test_tol = 1e-5
    iter_tol = 1e-8

    for test in 1:ntests
        A = randn(n, n)
        A = A + A'
        testMatrix(A, n, k, "Small random spd matrix test $test", test_tol=test_tol, iter_tol=iter_tol, verbose=false)
    end
end
