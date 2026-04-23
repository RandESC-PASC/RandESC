using Test
using RandESC
using Random
using LinearAlgebra

# jd_sketched always promotes its workspace to complex, so real_srtt (DCT-based)
# is incompatible regardless of input type. complex_srtt (FFT-based) works for both.
function sketch_types_for(T::Type)
    if T <: Real
        return ["real_gaussian", "real_srtt", "sparsesign", "sparsestack"]
    else
        return ["complex_gaussian", "complex_srtt", "sparsesign", "sparsestack"]
    end
end

const ORTH_METHODS = [:rgs, :rcgs, :rcgs2]
const JD_ORTH_METHODS = [:mgs, :mgs2, :qr]

# Check that each returned eigenpair (lambda[i], V[:,i]) satisfies the eigenvalue equation
# ‖A*v - λ*v‖ / ‖v‖ < res_tol.
function testResiduals(A, lambda, V, res_tol)
    k = length(lambda)
    for i in 1:k
        v = V[:, i]
        rnorm = norm(A * v - lambda[i] * v) / norm(v)
        @test rnorm < res_tol
    end
end


function testMatrix(A, n, k, testName; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    # compute true results
    ea = eigen(A)
    evals = ea.values
    evecs = ea.vectors
    evals = evals[1:k]
    evecs = evecs[:, 1:k]
    v0 = randn(eltype(A), n, k)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    # testing block randomized JD
    @testset "$testName: testing jd_sketched" begin
        V_rjdb, lambda_rjdb, history_rjdb = jd_sketched(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        @test eltype(V_rjdb) == expected_vec_type
        @test eltype(lambda_rjdb) == expected_val_type
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
        # jd_sketched converges in sketch norm; full residual may exceed iter_tol by a factor of ~2
        testResiduals(A, lambda_rjdb, V_rjdb, 2. * iter_tol)
    end

    # testing standard block JD
    @testset "$testName: testing jd" begin
        V_jdb, lambda_jdb, _ = jd(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)

        @test eltype(V_jdb) == expected_vec_type
        @test eltype(lambda_jdb) == expected_val_type
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
        testResiduals(A, lambda_jdb, V_jdb, iter_tol)
    end
end

function testJdOptions(A, n, k, testName; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    ea = eigen(A)
    evals = ea.values[1:k]
    v0 = randn(eltype(A), n, k)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    for orth_method in JD_ORTH_METHODS
        @testset "$testName: jd(orth_method=$orth_method)" begin
            V, lambda, _ = jd(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose,
                              orth_method=orth_method)
            @test eltype(V) == expected_vec_type
            @test eltype(lambda) == expected_val_type
            @test maximum(abs.(evals .- lambda)) < test_tol
            gram = V' * V
            @test maximum(abs.(gram - I)) < test_tol
            testResiduals(A, lambda, V, iter_tol)
        end
    end
end

function testJdSketchedOptions(A, n, k, testName; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    ea = eigen(A)
    evals = ea.values[1:k]
    v0 = randn(eltype(A), n, k)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    for sketch_type in sketch_types_for(eltype(A))
        for orth_method in ORTH_METHODS
            @testset "$testName: jd_sketched(sketch_type=$sketch_type, orth_method=$orth_method)" begin
                V, lambda, _ = jd_sketched(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose,
                                           sketch_type=sketch_type, orth_method=orth_method)
                @test eltype(V) == expected_vec_type
                @test eltype(lambda) == expected_val_type
                @test maximum(abs.(evals .- lambda)) < test_tol
                gram = V' * V
                @test maximum(abs.(gram - I)) < test_tol
                testResiduals(A, lambda, V, 2 * iter_tol)
            end
        end
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

# Test all orth_method options for jd, for each input type.
# Uses a single fixed matrix per type to keep runtime reasonable.
@testset "RandESC jd Options — Float64" begin
    Random.seed!(11111)
    n, k = 200, 5
    A = randn(Float64, n, n); A = A + A'
    testJdOptions(A, n, k, "Float64"; test_tol=1e-5, iter_tol=1e-8)
end

@testset "RandESC jd Options — ComplexF64" begin
    Random.seed!(22222)
    n, k = 200, 4
    A = randn(ComplexF64, n, n); A = A + A'
    testJdOptions(A, n, k, "ComplexF64"; test_tol=1e-5, iter_tol=1e-8)
end

@testset "RandESC jd Options — Float32" begin
    Random.seed!(33333)
    n, k = 200, 4
    A = randn(Float32, n, n); A = A + A'
    testJdOptions(A, n, k, "Float32"; test_tol=1e-2, iter_tol=1e-3)
end

@testset "RandESC jd Options — ComplexF32" begin
    Random.seed!(44444)
    n, k = 200, 4
    A = randn(ComplexF32, n, n); A = A + A'
    testJdOptions(A, n, k, "ComplexF32"; test_tol=1e-2, iter_tol=1e-3)
end

@testset "RandESC jd_sketched Options — Float64" begin
    Random.seed!(11111)
    n, k = 200, 5
    A = randn(Float64, n, n); A = A + A'
    testJdSketchedOptions(A, n, k, "Float64"; test_tol=1e-5, iter_tol=1e-8)
end

@testset "RandESC jd_sketched Options — ComplexF64" begin
    Random.seed!(22222)
    n, k = 200, 4
    A = randn(ComplexF64, n, n); A = A + A'
    testJdSketchedOptions(A, n, k, "ComplexF64"; test_tol=1e-5, iter_tol=1e-8)
end

@testset "RandESC jd_sketched Options — Float32" begin
    Random.seed!(33333)
    n, k = 200, 4
    A = randn(Float32, n, n); A = A + A'
    testJdSketchedOptions(A, n, k, "Float32"; test_tol=1e-2, iter_tol=1e-3)
end

@testset "RandESC jd_sketched Options — ComplexF32" begin
    Random.seed!(44444)
    n, k = 200, 4
    A = randn(ComplexF32, n, n); A = A + A'
    testJdSketchedOptions(A, n, k, "ComplexF32"; test_tol=1e-2, iter_tol=1e-3)
end