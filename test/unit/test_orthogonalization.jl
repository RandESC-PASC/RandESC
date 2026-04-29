using Test
using RandESC
using Random
using LinearAlgebra

const ORTH_TOL        = 1e-12   # orthonormality: Q'Q ≈ I
const SPAN_TOL        = 1e-12   # span check: residual outside Q
const SKETCH_ORTH_TOL = 1e-10   # Θ-orthonormality: SV'SV ≈ I
const SKETCH_SPAN_TOL = 1e-10   # span check in sketch space
const SKETCH_V_ORTH_TOL = 0.5  # approximate orthonormality of V_out (JL distortion ~√(p/s))

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

@testset "Sketch orthogonalization" begin
    Random.seed!(42)
    n, s, p = 200, 120, 10
    Theta = RandESC.sketch(n, s, "real_gaussian", randn(n))

    V_indep = randn(n, p)
    V_full  = hcat(V_indep, V_indep[:, 1], V_indep[:, 3], V_indep[:, 5])
    m = size(V_full, 2)

    for method in (:rgs, :rcgs, :rcgs2, :rqr)
        @testset "$method: mixed independent and duplicate columns" begin
            V0    = copy(V_full)
            V_out = similar(V0)
            SV_out = zeros(s, m)
            SV_buf = zeros(s, m)
            nact = RandESC._jd_theta_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)

            @test nact == p

            SQ = SV_out[:, 1:nact]
            Q  = V_out[:, 1:nact]
            @test SQ' * SQ ≈ I atol=SKETCH_ORTH_TOL
            v_orth_err = maximum(abs.(Q' * Q - I))
            # @info "V_out orth error" method v_orth_err
            @test v_orth_err < SKETCH_V_ORTH_TOL

            for i in 1:p
                sv   = Theta(V_indep[:, i:i])[:, 1]
                proj = SQ * (SQ' * sv)
                @test norm(sv - proj) < SKETCH_SPAN_TOL
            end
        end
    end

    @testset "full-rank input" begin
        for method in (:rgs, :rcgs, :rcgs2, :rqr)
            V0    = randn(n, p)
            V_out = similar(V0)
            SV_out = zeros(s, p)
            SV_buf = zeros(s, p)
            nact = RandESC._jd_theta_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)
            @test nact == p
            @test SV_out[:, 1:nact]' * SV_out[:, 1:nact] ≈ I atol=SKETCH_ORTH_TOL
            v_orth_err = maximum(abs.(V_out[:, 1:nact]' * V_out[:, 1:nact] - I))
            # @info "V_out orth error" method v_orth_err
            @test v_orth_err < SKETCH_V_ORTH_TOL
        end
    end

    @testset "all-duplicate input (rank 1)" begin
        v = randn(n)
        V = repeat(v, 1, 5)
        for method in (:rgs, :rcgs, :rcgs2, :rqr)
            V0    = copy(V)
            V_out = similar(V0)
            SV_out = zeros(s, 5)
            SV_buf = zeros(s, 5)
            nact = RandESC._jd_theta_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)
            @test nact == 1
            @test norm(SV_out[:, 1])^2 ≈ 1.0 atol=SKETCH_ORTH_TOL
        end
    end

    @testset "theta_orth_block_against!" begin
        for method in (:rgs, :rcgs, :rcgs2)
            Q0    = randn(n, p ÷ 2)
            V_out = similar(Q0)
            SV_out = zeros(s, p ÷ 2)
            SV_buf = zeros(s, p ÷ 2)
            RandESC._jd_theta_ortho!(V_out, SV_out, Q0, Theta, :rgs, SV_buf)
            Q = V_out[:, 1:p÷2]; SQ = SV_out[:, 1:p÷2]

            V1     = randn(n, p ÷ 2)
            SV1_buf = zeros(s, p ÷ 2)
            RandESC.theta_orth_block_against!(V1, Q, SQ, Theta, method, SV1_buf)

            @test norm(SQ' * Theta(V1)) < SKETCH_ORTH_TOL
        end
    end
end
