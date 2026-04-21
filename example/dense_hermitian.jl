using RandESC
using LinearAlgebra
using Random

Random.seed!(42)

n = 200   # matrix size
k = 10    # number of eigenpairs to compute

# Build a random dense Hermitian matrix
A = randn(ComplexF64, n, n)
A = A + A'

# Reference solution from Julia's dense eigensolver
ref = eigen(Hermitian(A))
lambda_ref = ref.values[1:k]

X0 = randn(ComplexF64, n, k)

# Standard block Jacobi-Davidson
V_jd, lambda_jd, history_jd = jd(A, X0; k=k, tol=1e-10, maxit=100, disp=false)
println("jd:      $(size(history_jd,1)) iters, $(Int(history_jd[end,3])) matvecs, max eigenvalue error = $(maximum(abs.(lambda_jd .- lambda_ref)))")

# Randomized (sketched) block Jacobi-Davidson
V_rjd, lambda_rjd, history_rjd = jd_sketched(A, X0; k=k, tol=1e-10, maxit=100, disp=false)
println("jd_sketched: $(size(history_rjd,1)) iters, $(Int(history_rjd[end,3])) matvecs, max eigenvalue error = $(maximum(abs.(lambda_rjd .- lambda_ref)))")
