using Test
using RandESC
using Random
using LinearAlgebra

const ORTH_TOL = 1e-12   # orthonormality: Q'Q ≈ I
const SPAN_TOL = 1e-12   # span check: residual outside Q

@testset "Standard orthogonalization" begin
    Random.seed!(42)
    n = 50   # ambient dimension
    p = 16   # number of independent vectors

    # Build a matrix with p independent columns followed by 3 duplicates (linear deps)
    V_indep = randn(n, p)
    # Append duplicates of columns 1, 3, 5 to create some linear dependencies
    V_full = hcat(V_indep, V_indep[:, 1], V_indep[:, 3], V_indep[:, 5])
    m = size(V_full, 2)

    for method in (:mgs, :mgs2, :qr)
        @testset "$method: mixed independent and duplicate columns" begin
            V = copy(V_full)
            buf = similar(V, n, m)
            nact = RandESC._jd_ortho!(V, m, method, buf)

            # Should recover exactly p independent vectors
            @test nact == p

            Q = V[:, 1:nact]

            # Columns must be orthonormal
            G = Q' * Q
            @test G ≈ I atol=ORTH_TOL

            # Span must match the original independent columns:
            # each original column should lie in span(Q)
            for i in 1:p
                v = V_indep[:, i]
                proj = Q * (Q' * v)
                @test norm(v - proj) < SPAN_TOL
            end
        end
    end

    @testset "full-rank input (no linear dependencies)" begin
        for method in (:mgs, :mgs2, :qr)
            V = randn(n, p)
            nact = RandESC._jd_ortho!(V, p, method)
            @test nact == p
            @test V[:, 1:nact]' * V[:, 1:nact] ≈ I atol=ORTH_TOL
        end
    end

    @testset "all-duplicate input (rank 1)" begin
        v = randn(n)
        V = repeat(v, 1, 5)   # 5 identical columns
        for method in (:mgs, :mgs2, :qr)
            Vc = copy(V)
            nact = RandESC._jd_ortho!(Vc, 5, method)
            @test nact == 1
            q = Vc[:, 1]
            @test norm(q) ≈ 1.0 atol=ORTH_TOL
        end
    end
end
