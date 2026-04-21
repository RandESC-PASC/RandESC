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
energy_tol = 1e-5
max_scf_iter = 10

println("=== Standard block Jacobi-Davidson ===")
scfres_jd = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=false))
println(scfres_jd.energies)

energy_jd = scfres_jd.energies.total
iter_jd   = scfres_jd.n_iter
@assert abs(energy_jd - expected_energy) < energy_tol "JD total energy $energy_jd differs from expected $expected_energy by $(abs(energy_jd - expected_energy))"
@assert iter_jd <= max_scf_iter "JD used $iter_jd SCF iterations, expected <= $max_scf_iter"
println("JD: energy=$energy_jd, n_iter=$iter_jd — PASSED")

println("\n=== Randomized block Jacobi-Davidson ===")
scfres_rjd = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=true))
println(scfres_rjd.energies)

energy_rjd = scfres_rjd.energies.total
iter_rjd   = scfres_rjd.n_iter
@assert abs(energy_rjd - expected_energy) < energy_tol "RJD total energy $energy_rjd differs from expected $expected_energy by $(abs(energy_rjd - expected_energy))"
@assert iter_rjd <= max_scf_iter "RJD used $iter_rjd SCF iterations, expected <= $max_scf_iter"
println("RJD: energy=$energy_rjd, n_iter=$iter_rjd — PASSED")
