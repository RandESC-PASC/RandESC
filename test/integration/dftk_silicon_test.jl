using Test
using DFTK
using PseudoPotentialData
using RandESC

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si        = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model = model_DFT(lattice, atoms, positions; functionals=LDA())
basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2])

function make_eigensolver(; useRandomization)
    function eigensolver(A, X0; prec=nothing, maxiter, tol, kwargs...)
        function precond_preparation(M, X)
            DFTK.precondprep!(M, X)
        end
        return randESCSolver(A, X0, size(X0, 1), size(X0, 2);
                             preconditioner=prec,
                             precond_preparator=precond_preparation,
                             maxiter=maxiter,
                             tol=tol,
                             useRandomization=useRandomization,
                             verbose=false)
    end
end

expected_energy = -8.42886007706835
energy_tol = 1e-7
max_scf_iter = 10

@testset "DFTK Silicon — standard JD" begin
    scfres = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=false))
    @test abs(scfres.energies.total - expected_energy) < energy_tol
    @test scfres.n_iter <= max_scf_iter
end

@testset "DFTK Silicon — randomized JD" begin
    scfres = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=true))
    @test abs(scfres.energies.total - expected_energy) < energy_tol
    @test scfres.n_iter <= max_scf_iter
end
