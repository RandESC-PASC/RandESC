using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using Printf

# set random seed for reproducibility
# using Random
# Random.seed!(1234)

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
        # println("computing matrix")
        # H = Array(A)
        # println("writing matrix to disk ")
        # open("hamiltonian.bin", "w") do io
        #     write(io, size(H, 1))
        #     write(io, size(H, 2))
        #     write(io, H)
        # end
        # println("done")

        # println("writing preconditioner")
        # precond_vec = (prec.kin .+ prec.default_shift)
        # println("size of preconditioner ", size(precond_vec, 1))
        # open("preconditioner.bin", "w") do io
        #     write(io, size(precond_vec, 1))
        #     write(io, precond_vec)
        # end

        println("computing matrix")
        println("size of matrix ", size(X0))
        H = Array(A)
        precond_vec = (prec.kin .+ prec.default_shift)
        open("preconditioner_vector.txt", "w") do io
            for i in 1:length(precond_vec)
                print(io, @sprintf("%.15e\n", precond_vec[i]))
            end
        end
        println("Writing hamiltonian to disk as hamiltonian.txt ... ", H[1, 1])
        open("hamiltonian.txt", "w") do io
            for i in 1:size(H, 1)
                for j in 1:size(H, 2)
                    # print(io, @sprintf("%.15e ", H[i, j]))
                    print(io, @sprintf("%.15e%+.15ei ", real(H[i, j]), imag(H[i, j])))
                end
                print(io, "\n")
            end
        end
        println("Done.")
        println("writing initial guess to disk as initial.txt")
        open("initial.txt", "w") do io
            for i in 1:size(X0, 1)
                for j in 1:size(X0, 2)
                    print(io, @sprintf("%.15e%+.15ei ", real(X0[i, j]), imag(X0[i, j])))
                end
                print(io, "\n")
            end
        end
        println("done writing")
    end

    return randESCSolver(A, size(X0, 1), size(X0, 2); X0=X0, preconditioner=prec, maxiter=maxiter, tol=tol, useRandomization=false, method="JD", cleanEigenvectors=false, verbose=false, precond_preparator=precond_preparation, normA=ecut)
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
family = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
# family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
for i in 1:nat
    # pp = ElementPsp(Symbol(species[i]), PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
    pp = ElementPsp(Symbol(species[i]), family)
    push!(atom_psps, pp)
end


invlat = inv(lattice)
positionvecs = [ invlat * positions[:, i] for i in 1:nat]


model = model_DFT(lattice, atom_psps, positionvecs; spin_polarization=:none, functionals=PBE(), temperature=0.01)
# badbasis = PlaneWaveBasis(model, Ecut=4,  kgrid=[2, 2, 1])
# println("starting bad basis")
# compres = self_consistent_field(badbasis)
# compres = self_consistent_field(badbasis, eigensolver = my_eig_solver);
println("starting good basis")
basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=[2, 2, 1])
DFTK.reset_timer!(DFTK.timer)
println("why am i so slow?")
# scfres = self_consistent_field(basis)#, eigensolver = my_eig_solver);
scfres = self_consistent_field(basis, eigensolver = my_eig_solver);
println(DFTK.timer)


# # Print occupations
# println("Occupations of Kohn-Sham states:")
# for (i, occ) in enumerate(scfres.occupation)
#     println("State $i: occupation = $occ")
# end


# DFTK.reset_timer!(DFTK.timer)
# # scfres = self_consistent_field(basis)#, eigensolver = my_eig_solver);
# scfres = self_consistent_field(basis, eigensolver = my_eig_solver);
# println(DFTK.timer)


# # Print occupations
# println("Occupations of Kohn-Sham states:")
# for (i, occ) in enumerate(scfres.occupation)
#     println("State $i: occupation = $occ")
# end
