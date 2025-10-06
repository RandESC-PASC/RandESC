
include("sketch.jl")
include("IRA.jl")
include("RIRA.jl")
include("LOBPCG.jl")

using DelimitedFiles

n = 50

k = 16
m = max(2k, k + 20) # restarted Arnoldi subspace dimension

# example spd matrix
A = randn(n,n)
A = A' + A
# example hermitian matrix
B = randn(n,n) + im*randn(n,n)
B = (B + B') / 2

# V, D, ritz, info = rand_ira(A, k, m; verbose=false)
# println("number of matrix-vector products: ", info.mvps)

# script should be run with julia script.jl filename
filename = ARGS[1]
println("filename: ", filename)

# matrix is stored in text in filename
A = readdlm(filename)

V, D, ritz, info = rand_ira(A, k, m; verbose=true)
println("number of matrix-vector products: ", info.mvps)

# matrix is stored in text in filename
A = readdlm(filename)

n = size(A, 1)
println("size of A: ", n)
# and now for lobpcg
X, Lambda, info = lobpcg(A, n, k; verbosity=1)
println("number of matrix-vector products: ", info.mvps)    