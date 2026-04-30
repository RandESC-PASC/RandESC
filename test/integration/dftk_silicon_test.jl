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

function make_eigensolver(; use_randomization)
    function eigensolver(A, X0; prec=nothing, maxiter, tol, kwargs...)
        function precond_preparation(M, X)
            DFTK.precondprep!(M, X)
        end
        return randESCSolver(A, X0, size(X0, 1), size(X0, 2);
                             preconditioner=prec,
                             precond_preparator=precond_preparation,
                             maxiter=maxiter,
                             tol=tol,
                             use_randomization=use_randomization,
                             verbose=false)
    end
end

energy_tol = 1e-7
scf_iter_margin = 5

# Reference: DFTK's own eigensolver, converged to energy_tol.
# Computed on the fly so the test is robust against DFTK internal changes.
ref_scfres = self_consistent_field(basis)
ref_energy = ref_scfres.energies.total
ref_iter   = ref_scfres.n_iter

@testset "DFTK Silicon — standard JD" begin
    scfres = self_consistent_field(basis; eigensolver=make_eigensolver(; use_randomization=false))
    @test abs(scfres.energies.total - ref_energy) < energy_tol
    @test abs(scfres.n_iter - ref_iter) <= scf_iter_margin
end

@testset "DFTK Silicon — randomized JD" begin
    scfres = self_consistent_field(basis; eigensolver=make_eigensolver(; use_randomization=true))
    @test abs(scfres.energies.total - ref_energy) < energy_tol
    @test abs(scfres.n_iter - ref_iter) <= scf_iter_margin
end
