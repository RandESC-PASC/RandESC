
include("sketch.jl")
include("IRA.jl")
include("RIRA.jl")
include("LOBPCG.jl")

n = 50

k = 3
m = min(2 * k, 20)

# example spd matrix
A = randn(n,n)
A = A' + A
# example hermitian matrix
B = randn(n,n) + im*randn(n,n)
B = (B + B') / 2

V, D, ritz, info = rand_ira(A, k, m; verbose=true)
println("number of matrix-vector products: ", info.mvps)
