using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC

# precondprep!(P, X) = P
ecut = 30

function my_eig_solver(A, X0; prec = nothing, maxiter, tol, kwargs...)

    # define preconditioner preparation function
    function precond_preparation(M, X)
        precondprep!(M, X)
        # DFTK.Eigen.Preconditioners.precondprep(M, X)
    end

    writeOperator = false
    if writeOperator
        println("computing matrix")
        H = Array(A)
        println("writing matrix to disk ")
        open("hamiltonian.bin", "w") do io
            write(io, size(H, 1))
            write(io, size(H, 2))
            write(io, H)
        end
        println("done")

        println("writing preconditioner")
        precond_vec = (prec.kin .+ prec.default_shift)
        println("size of preconditioner ", size(precond_vec, 1))
        open("preconditioner.bin", "w") do io
            write(io, size(precond_vec, 1))
            write(io, precond_vec)
        end
    end

    return randESCSolver(A, size(X0, 1), size(X0, 2); X0=X0, preconditioner=prec, maxiter=maxiter, tol=tol, useRandomization=false, method="LOBPCG", cleanEigenvectors=false, verbose=true, precond_preparator=precond_preparation, normA=ecut)
end


filename = "input.extxyz"
# check if command line argument is given
if length(ARGS) > 0
    filename = ARGS[1]
end

structure = read_frame(filename)

ang2bohr = 1.8897261249935897
positions = structure["arrays"]["pos"] * ang2bohr
latticeP = structure["cell"] * ang2bohr
lattice = latticeP'
nat = structure["N_atoms"]
species = structure["arrays"]["species"]

print("Read structure with $nat atoms from $filename\n")

atom_psps = ElementPsp[]
for i in 1:nat
    pp = ElementPsp(Symbol(species[i]), PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
    push!(atom_psps, pp)
end


invlat = inv(lattice)
positionvecs = [ invlat * positions[:, i] for i in 1:nat]


model = model_DFT(lattice, atom_psps, positionvecs; functionals=PBE(), temperature=0.01)
basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=[1, 1, 1], kshift=[0, 0, 0])
DFTK.reset_timer!(DFTK.timer)
# scfres = self_consistent_field(basis)#, eigensolver = my_eig_solver);
scfres = self_consistent_field(basis, eigensolver = my_eig_solver);
println(DFTK.timer)
