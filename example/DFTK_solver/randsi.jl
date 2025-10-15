
# Very basic setup, useful for testing
using DFTK
using PseudoPotentialData
using Arpack
using LinearAlgebra
using LinearMaps
using KrylovKit
using Printf
using RandESC

precondprep!(P, X) = P
ecut = 50

function my_eig_solver(A, X0; prec = nothing, maxiter, tol, kwargs...)

    # define preconditioner preparation function
    function precond_preparation(M, X)
        precondprep!(M, X)
        # DFTK.Eigen.Preconditioners.precondprep(M, X)
    end

    return randESCSolver(A, size(X0, 1), size(X0, 2); X0=X0, preconditioner=prec, maxiter=maxiter*10, tol=tol, useRandomization=false, method="LOBPCG", cleanEigenvectors=false, verbose=false, precond_preparator=precond_preparation, normA=ecut)
end

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model = model_DFT(lattice, atoms, positions; functionals=LDA())
basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=[1, 1, 1])

DFTK.reset_timer!(DFTK.timer)
# scfres = self_consistent_field(basis)
scfres = self_consistent_field(basis, eigensolver = my_eig_solver);
# scfres = self_consistent_field(basis, eigensolver = my_eig_solver);
# scfres = self_consistent_field(basis, eigensolver = lobpcg_solver);
println(DFTK.timer)