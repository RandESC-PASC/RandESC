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
# A = randn(n,n)
# A = A' + A
# # example hermitian matrix
# B = randn(n,n) + im*randn(n,n)
# B = (B + B') / 2

# V, D, ritz, info = rand_ira(A, k, m; verbose=false)
# println("number of matrix-vector products: ", info.mvps)

# script should be run with julia script.jl filename
filename = ARGS[1]
# println("filename: ", filename)

# matrix is stored in text in filename
A = readdlm(filename)
n = size(A, 1)

V, D, ritz, info = rand_ira(A, n, k ; verbose=true)
V, D = sketched_to_fully(A, V)

v_lobpcg, lambda, info = lobpcg(A, n, k; verbosity=1)
println("number of matrix-vector products: ", info.mvps)

# normalize eigenvectors
for i in 1:k
    V[:, i] /= norm(V[:, i])
end

println("Gram matrix of evecs rand_ira:")
gram_rand_ira = round.(real(V' * V); digits=2)
for i in 1:k
    println(gram_rand_ira[i, :])
end

# # matrix is stored in text in filename
# A = readdlm(filename)

Random.seed!(98423598)
n = size(A, 1)
println("size of A: ", n)
# and now for lobpcg
X, Lambda, info = lobpcg(A, n, k; verbosity=1, normA=15)
println("number of matrix-vector products: ", info.mvps)

# and now the interface with

Random.seed!(98423598)
lambda_inter, X_inter, residual_norms_inter, n_iter_inter, converged_inter, n_matvecs_inter  = randESCSolver(A, n, k; method="LOBPCG", useRandomization=false, cleanEigenvectors=false, maxiter=300, tol=1e-8, verbose=true)

# print comparison of eigenvalues from rand_ira, and randESCSolver with method="lobpcg"
println("Eigenvalues from rand_ira:")
println(diag(D))
println("Eigenvalues from lobpcg:")
println(lambda)

println("Eigenvalues from randESCSolver with method=\"LOBPCG\":")
println(lambda_inter)
