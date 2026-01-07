using RandESC
using DelimitedFiles
using Random
using LinearAlgebra
# Random.seed!(312634)
Random.seed!(98423598)

n = 50

k = 7
m = max(2k, k + 20) # restarted Arnoldi subspace dimension

# # example spd matrix
A = randn(n,n) + im*randn(n,n)
A = A' + A
# # example hermitian matrix
# B = randn(n,n) + im*randn(n,n)
# B = (B + B') / 2

# V, D, ritz, info = rand_ira(A, k, m; verbose=false)
# println("number of matrix-vector products: ", info.mvps)

# script should be run with julia script.jl filename
# filename = ARGS[1]
# # println("filename: ", filename)

# # matrix is stored in text in filename
# A = readdlm(filename)
# n = size(A, 1)

# # matrix is stored in text in filename
# A = readdlm(filename)

Random.seed!(98423598)
n = size(A, 1)
println("size of A: ", n)
# and now for lobpcg
# X, Lambda, info = lobpcg(A, n, k; verbosity=1, normA=15)
# println("number of matrix-vector products: ", info.mvps)

# and now the interface with

Random.seed!(98423598)
lambda_inter, X_inter, residual_norms_inter, n_iter_inter, converged_inter, n_matvecs_inter  = randESCSolver(A, n, k; method="LOBPCG", useRandomization=false, cleanEigenvectors=false, maxiter=300, tol=1e-8, verbose=true)

# Test JD method
println("\n=== Testing Jacobi-Davidson ===")
V_jd, lambda_jd, history_jd = jdsym(A; k=k, tol=1e-8, maxit=300, disp=true)
println("JD converged in $(size(history_jd, 1)) iterations")
println("JD number of matrix-vector products: ", Int(history_jd[end, 3]))

# Test randomized JD method
println("\n=== Testing Randomized Jacobi-Davidson ===")
V_rjd, lambda_rjd, history_rjd = jdsym_rand(A; k=k, tol=1e-8, maxit=300, disp=true)
V_rjd, lambda_rjd = sketched_to_fully(A, V_rjd)
println("randJD converged in $(size(history_rjd, 1)) iterations")
println("randJD number of matrix-vector products: ", Int(history_rjd[end, 3]))

# Test JD through interface
println("\n=== Testing JD through interface ===")
result_jd = randESCSolver(A, n, k; method="JD", useRandomization=false, maxiter=300, tol=1e-8, verbose=true)
result_rjd = randESCSolver(A, n, k; method="JD", useRandomization=true, cleanEigenvectors=true, maxiter=300, tol=1e-8, verbose=true)

# print comparison of eigenvalues from rand_ira, and randESCSolver with method="lobpcg"
# println("\n=== Eigenvalue Comparison ===")
# println("Eigenvalues from rand_ira:")
# println(diag(D))
# println("\nEigenvalues from lobpcg:")
# println(lambda)
println("\nEigenvalues from randESCSolver with method=\"LOBPCG\":")
println(lambda_inter)
println("\nEigenvalues from JD:")
println(lambda_jd)
println("\nEigenvalues from randJD:")
println(lambda_rjd)
println("\nEigenvalues from interface JD:")
println(result_jd.λ)
println("\nEigenvalues from interface randJD:")
println(result_rjd.λ)
