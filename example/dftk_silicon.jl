using DFTK
using PseudoPotentialData
using RandESC

# Silicon unit cell (FCC, two atoms per cell)
a = 10.26  # lattice constant in Bohr
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

println("=== Standard block Jacobi-Davidson ===")
scfres_jd = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=false))
println(scfres_jd.energies)

println("\n=== Randomized block Jacobi-Davidson ===")
scfres_rjd = self_consistent_field(basis; eigensolver=make_eigensolver(; useRandomization=true))
println(scfres_rjd.energies)
