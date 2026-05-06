using Test
using RandESC
using Random
using LinearAlgebra

### This file defines all functions necessary for the unit testing of symmetric and Hermitian
### random matrices. The main function run_symmetric_hermitian_random_matrix_tests(; template)
### launches all tests for a given template array, determining on which architectre the tests
### should run (CPU, NVIDIA GPU, AMD GPU). The function is called with approriate template
### from the cpu.jl, cuda.jl, and amdgpu.jl test files.

function sketch_types_for(T::Type)
    if T <: Real
        return ["real_gaussian", "real_srtt", "sparsesign", "sparsestack"]
    else
        return ["complex_gaussian", "complex_srtt", "sparsesign", "sparsestack"]
    end
end

function orth_methods()
    return [:rcgs, :rcgs2, :rqr]
end

function jd_orth_methods()
    return [:mgs, :mgs2, :qr]
end

# Check that each returned eigenpair (lambda[i], V[:,i]) satisfies the eigenvalue equation
# ‖A*v - λ*v‖ / ‖v‖ < res_tol.
function test_residuals(A, lambda, V, res_tol)
    k = length(lambda)
    for i in 1:k
        v = V[:, i]
        rnorm = norm(A * v - lambda[i] * v) / norm(v)
        @test rnorm < res_tol
    end
end

function test_matrix(A, n, k, test_name; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    # compute true results
    A_cpu = RandESC.to_cpu(A)
    ea = eigen(A_cpu)
    evals = ea.values
    evecs = ea.vectors
    evals = evals[1:k]
    evecs = evecs[:, 1:k]
    v0 = RandESC.random_matrix(eltype(A), n, k, A)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    # testing block randomized JD
    @testset "$test_name: testing jd_sketched" begin
        V_rjdb, lambda_rjdb, history_rjdb = jd_sketched(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)
        V_rjdb = RandESC.to_cpu(V_rjdb)
        lambda_rjdb = RandESC.to_cpu(lambda_rjdb)

        @test eltype(V_rjdb) == expected_vec_type
        @test eltype(lambda_rjdb) == expected_val_type
        rjdb_passed = maximum(abs.(evals .- lambda_rjdb)) < test_tol
        if !rjdb_passed
            @warn "$test_name: jd_sketched eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_rjdb)))"
        end
        @test rjdb_passed
        gram_rjdb = V_rjdb' * V_rjdb
        rjdb_orth_passed = maximum(abs.(gram_rjdb - I)) < test_tol
        if !rjdb_orth_passed
            @warn "$test_name: jd_sketched eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_rjdb - I)))"
        end
        @test rjdb_orth_passed
        # jd_sketched converges in sketch norm; full residual may exceed iter_tol by a factor of ~2
        test_residuals(A_cpu, lambda_rjdb, V_rjdb, 2. * iter_tol)
    end

    # testing standard block JD
    @testset "$test_name: testing jd" begin
        V_jdb, lambda_jdb, _ = jd(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose)
        V_jdb = RandESC.to_cpu(V_jdb)
        lambda_jdb = RandESC.to_cpu(lambda_jdb)

        @test eltype(V_jdb) == expected_vec_type
        @test eltype(lambda_jdb) == expected_val_type
        jdb_passed = maximum(abs.(evals .- lambda_jdb)) < test_tol
        if !jdb_passed
            @warn "$test_name: jd eigenvalues did not match! Max abs error: $(maximum(abs.(evals .- lambda_jdb)))"
        end
        @test jdb_passed
        gram_jdb = V_jdb' * V_jdb
        jdb_orth_passed = maximum(abs.(gram_jdb - I)) < test_tol
        if !jdb_orth_passed
            @warn "$test_name: jd eigenvectors are not orthogonal! Max abs error: $(maximum(abs.(gram_jdb - I)))"
        end
        @test jdb_orth_passed
        test_residuals(A_cpu, lambda_jdb, V_jdb, iter_tol)
    end
end

function test_jd_options(A, n, k, test_name; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    A_cpu = RandESC.to_cpu(A)
    ea = eigen(A_cpu)
    evals = ea.values[1:k]
    v0 = RandESC.random_matrix(eltype(A), n, k, A)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    for orth_method in jd_orth_methods()
        @testset "$test_name: jd(orth_method=$orth_method)" begin
            V, lambda, _ = jd(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose,
                              orth_method=orth_method)
            V = RandESC.to_cpu(V)
            lambda = RandESC.to_cpu(lambda)
            @test eltype(V) == expected_vec_type
            @test eltype(lambda) == expected_val_type
            @test maximum(abs.(evals .- lambda)) < test_tol
            gram = V' * V
            @test maximum(abs.(gram - I)) < test_tol
            test_residuals(A_cpu, lambda, V, iter_tol)
        end
    end
end

function test_jd_sketched_options(A, n, k, test_name; test_tol=1e-5, iter_tol=1e-8, verbose=false, maxiter=1000)
    A_cpu = RandESC.to_cpu(A)
    ea = eigen(A_cpu)
    evals = ea.values[1:k]
    v0 = RandESC.random_matrix(eltype(A), n, k, A)

    expected_vec_type = eltype(A)
    expected_val_type = real(eltype(A))

    for sketch_type in sketch_types_for(eltype(A))
        for orth_method in orth_methods()
            @testset "$test_name: jd_sketched(sketch_type=$sketch_type, orth_method=$orth_method)" begin
                V, lambda, _ = jd_sketched(A, v0; k=k, tol=iter_tol, maxit=maxiter, disp=verbose,
                                           sketch_type=sketch_type, orth_method=orth_method)
                V = RandESC.to_cpu(V)
                lambda = RandESC.to_cpu(lambda)
                @test eltype(V) == expected_vec_type
                @test eltype(lambda) == expected_val_type
                @test maximum(abs.(evals .- lambda)) < test_tol
                gram = V' * V
                @test maximum(abs.(gram - I)) < test_tol
                test_residuals(A_cpu, lambda, V, 2 * iter_tol)
            end
        end
    end
end


# Small test with random spd matrix
function random_spd_test(T::Type; ntests=7, test_tol=1e-5, iter_tol=1e-8, seed=98423598, template=nothing)
    Random.seed!(seed)

    for test in 1:ntests
        n = (test + 3) * (test + 3)
        k = max(1, ceil(Int, 0.05 * n))
        A = RandESC.random_matrix(T, n, n, template)
        A = A + A'
        test_matrix(A, n, k, "Small random spd matrix test $test", test_tol=test_tol, iter_tol=iter_tol, verbose=false)
    end
end

# Testing a given set of jd orth_method options. Uses a single fixed matrix per type to keep runtime reasonable.
function jd_options_test(T::Type; n=200, k=5, test_tol=1e-5, iter_tol=1e-8,
                             seed=12345, template=nothing)
    Random.seed!(seed)
    A = RandESC.random_matrix(T, n, n, template)
    A = A + A'
    test_jd_options(A, n, k, string(T); test_tol=test_tol, iter_tol=iter_tol)
end

# Testing a given set of sketch options. Uses a single fixed matrix per type to keep runtime reasonable.
function sketch_options_test(T::Type; n=200, k=5, test_tol=1e-5, iter_tol=1e-8,
                                 seed=12345, template=nothing)
    Random.seed!(seed)
    A = RandESC.random_matrix(T, n, n, template)
    A = A + A'
    test_jd_sketched_options(A, n, k, string(T); test_tol=test_tol, iter_tol=iter_tol)
end

function run_symmetric_hermitian_random_matrix_tests(; template=nothing)

    ### Testing with random spd matrices of various types
    @testset "RandESC Symmetric Eigenvalue Solver Tests" begin
        random_spd_test(Float64; ntests=7, test_tol=1e-5, iter_tol=1e-8, template=template)
    end

    @testset "RandESC Hermitian Eigenvalue Solver Tests" begin
        random_spd_test(ComplexF64, ntests=7, test_tol=1e-5, iter_tol=1e-8, template=template)
    end

    # Tolerances are pretty loose for single precision tests, otherwise they fail
    @testset "RandESC Symmetric Single Precision Eigenvalue Solver Tests" begin
        random_spd_test(Float32; ntests=8, test_tol=1e-3, iter_tol=1e-3, template=template)
    end

    # Tolerances are pretty loose for single precision tests, otherwise they fail
    @testset "RandESC Hermitian Single Precision Eigenvalue Solver Tests" begin
        random_spd_test(ComplexF32, ntests=8, test_tol=1e-3, iter_tol=1e-3, template=template)
    end

    ### Test all orth_method options for jd, for each input type.
    @testset "RandESC jd Options — Float64" begin
        jd_options_test(Float64; n=200, k=5, test_tol=1e-5, iter_tol=1e-8,
                        seed=11111, template=template)
    end

    @testset "RandESC jd Options — ComplexF64" begin
        jd_options_test(ComplexF64; n=200, k=4, test_tol=1e-5, iter_tol=1e-8,
                        seed=22222, template=template)
    end

    @testset "RandESC jd Options — Float32" begin
        jd_options_test(Float32; n=200, k=4, test_tol=1e-2, iter_tol=1e-3,
                        seed=33333, template=template)
    end

    @testset "RandESC jd Options — ComplexF32" begin
        jd_options_test(ComplexF32; n=200, k=4, test_tol=1e-2, iter_tol=1e-3,
                        seed=44444, template=template)
    end

    ### Test all sketch_type and orth_method combinations for jd_sketched, for each input type.
    @testset "RandESC jd_sketched Options — Float64" begin
        sketch_options_test(Float64; n=200, k=5, test_tol=1e-5, iter_tol=1e-8,
                            seed=11111, template=template)
    end

    @testset "RandESC jd_sketched Options — ComplexF64" begin
        sketch_options_test(ComplexF64; n=200, k=4, test_tol=1e-5, iter_tol=1e-8,
                            seed=22222, template=template)
    end

    @testset "RandESC jd_sketched Options — Float32" begin
        sketch_options_test(Float32; n=200, k=4, test_tol=1e-2, iter_tol=1e-3,
                            seed=33333, template=template)
    end

    @testset "RandESC jd_sketched Options — ComplexF32" begin
        sketch_options_test(ComplexF32; n=200, k=4, test_tol=1e-2, iter_tol=1e-3,
                            seed=44444, template=template)
    end
end
