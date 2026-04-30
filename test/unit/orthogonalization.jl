using Test
using RandESC
using Random
using LinearAlgebra

const ORTH_TOL          = 1e-12   # orthonormality: Q'Q ≈ I
const SPAN_TOL          = 1e-12   # span check: residual outside Q
const SKETCH_ORTH_TOL   = 1e-10   # Θ-orthonormality: SV'SV ≈ I
const SKETCH_SPAN_TOL   = 1e-10   # span check in sketch space
const SKETCH_V_ORTH_TOL = 0.5    # approximate orthonormality of V_out (JL distortion ~√(p/s))

const STD_ORTH_METHODS    = (:mgs, :mgs2, :qr)
const SKETCH_ORTH_METHODS = (:rcgs, :rcgs2, :rqr)

function run_orthogonalization_tests(; template=nothing)
    T = Float64

    @testset "Standard orthogonalization" begin
        Random.seed!(42)
        n = 50   # ambient dimension
        p = 16   # number of independent vectors

        # Build a matrix with p independent columns followed by 3 duplicates (linear deps)
        V_indep_cpu = randn(T, n, p)
        V_full = RandESC.to_device(
            hcat(V_indep_cpu, V_indep_cpu[:, 1], V_indep_cpu[:, 3], V_indep_cpu[:, 5]),
            template)
        m = size(V_full, 2)

        for method in STD_ORTH_METHODS
            @testset "$method: mixed independent and duplicate columns" begin
                V = copy(V_full)
                buf = similar(V, n, m)
                nact = RandESC._ortho!(V, m, method, buf)

                @test nact == p

                Q = RandESC.to_cpu(V[:, 1:nact])

                @test Q' * Q ≈ I atol=ORTH_TOL

                for i in 1:p
                    v = V_indep_cpu[:, i]
                    proj = Q * (Q' * v)
                    @test norm(v - proj) < SPAN_TOL
                end
            end
        end

        @testset "full-rank input (no linear dependencies)" begin
            for method in STD_ORTH_METHODS
                V = RandESC.to_device(randn(T, n, p), template)
                nact = RandESC._ortho!(V, p, method)
                @test nact == p
                Q = RandESC.to_cpu(V[:, 1:nact])
                @test Q' * Q ≈ I atol=ORTH_TOL
            end
        end

        @testset "all-duplicate input (rank 1)" begin
            v = randn(T, n)
            V_cpu = repeat(v, 1, 5)
            for method in STD_ORTH_METHODS
                V = RandESC.to_device(copy(V_cpu), template)
                nact = RandESC._ortho!(V, 5, method)
                @test nact == 1
                q = RandESC.to_cpu(V[:, 1])
                @test norm(q) ≈ 1.0 atol=ORTH_TOL
            end
        end
    end

    @testset "Sketch orthogonalization" begin
        Random.seed!(42)
        n, s, p = 200, 120, 10
        Theta = RandESC.sketch(n, s, "real_gaussian", RandESC.to_device(randn(T, n), template))

        V_indep_cpu = randn(T, n, p)
        V_indep = RandESC.to_device(V_indep_cpu, template)
        V_full = RandESC.to_device(
            hcat(V_indep_cpu, V_indep_cpu[:, 1], V_indep_cpu[:, 3], V_indep_cpu[:, 5]),
            template)
        m = size(V_full, 2)

        for method in SKETCH_ORTH_METHODS
            @testset "$method: mixed independent and duplicate columns" begin
                V0     = copy(V_full)
                V_out  = similar(V0)
                SV_out = fill!(similar(V0, s, m), 0)
                SV_buf = fill!(similar(V0, s, m), 0)
                nact = RandESC._sketch_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)

                @test nact == p

                SQ = RandESC.to_cpu(SV_out[:, 1:nact])
                Q  = RandESC.to_cpu(V_out[:, 1:nact])
                @test SQ' * SQ ≈ I atol=SKETCH_ORTH_TOL
                @test maximum(abs.(Q' * Q - I)) < SKETCH_V_ORTH_TOL

                for i in 1:p
                    sv   = RandESC.to_cpu(Theta(V_indep[:, i:i]))[:, 1]
                    proj = SQ * (SQ' * sv)
                    @test norm(sv - proj) < SKETCH_SPAN_TOL
                end
            end
        end

        @testset "full-rank input" begin
            for method in SKETCH_ORTH_METHODS
                V0     = RandESC.to_device(randn(T, n, p), template)
                V_out  = similar(V0)
                SV_out = fill!(similar(V0, s, p), 0)
                SV_buf = fill!(similar(V0, s, p), 0)
                nact = RandESC._sketch_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)
                @test nact == p
                SQ = RandESC.to_cpu(SV_out[:, 1:nact])
                @test SQ' * SQ ≈ I atol=SKETCH_ORTH_TOL
                Q = RandESC.to_cpu(V_out[:, 1:nact])
                @test maximum(abs.(Q' * Q - I)) < SKETCH_V_ORTH_TOL
            end
        end

        @testset "all-duplicate input (rank 1)" begin
            v = randn(T, n)
            V_cpu = repeat(v, 1, 5)
            for method in SKETCH_ORTH_METHODS
                V0     = RandESC.to_device(copy(V_cpu), template)
                V_out  = similar(V0)
                SV_out = fill!(similar(V0, s, 5), 0)
                SV_buf = fill!(similar(V0, s, 5), 0)
                nact = RandESC._sketch_ortho!(V_out, SV_out, V0, Theta, method, SV_buf)
                @test nact == 1
                @test norm(RandESC.to_cpu(SV_out[:, 1]))^2 ≈ 1.0 atol=SKETCH_ORTH_TOL
            end
        end

        @testset "_sketch_deflate!" begin
            for method in SKETCH_ORTH_METHODS
                Q0     = RandESC.to_device(randn(T, n, p ÷ 2), template)
                V_out  = similar(Q0)
                SV_out = fill!(similar(Q0, s, p ÷ 2), 0)
                SV_buf = fill!(similar(Q0, s, p ÷ 2), 0)
                RandESC._sketch_ortho!(V_out, SV_out, Q0, Theta, :rcgs, SV_buf)
                Q = V_out[:, 1:p÷2]; SQ = SV_out[:, 1:p÷2]

                V1      = RandESC.to_device(randn(T, n, p ÷ 2), template)
                SV1_buf = fill!(similar(V1, s, p ÷ 2), 0)
                RandESC._sketch_deflate!(V1, Q, SQ, Theta, method, SV1_buf)

                @test norm(RandESC.to_cpu(SQ)' * RandESC.to_cpu(Theta(V1))) < SKETCH_ORTH_TOL
            end
        end
    end
end
