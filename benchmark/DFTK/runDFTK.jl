using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC

# Parse CLI arguments: [--gpu] [--ecut=N] [--kgrid=N,N,N] [filename]
use_gpu = "--gpu" in ARGS
get_flag(prefix) = let m = match(Regex("^$prefix=(.+)"), something(findfirst(a -> startswith(a, "$prefix="), ARGS) |> i -> i === nothing ? nothing : ARGS[i], ""))
    m === nothing ? nothing : m.captures[1]
end
ecut_arg  = get_flag("--ecut")
kgrid_arg = get_flag("--kgrid")
ecut  = ecut_arg  === nothing ? 30 : parse(Int, ecut_arg)
kgrid = kgrid_arg === nothing ? [2, 2, 2] : parse.(Int, split(kgrid_arg, ","))
filename_args = filter(a -> !startswith(a, "--"), ARGS)
filename = length(filename_args) > 0 ? filename_args[1] : "structures/si.extxyz"

# Read structure
structure = read_frame(filename)
ang2bohr = 1.8897261249935897
positions = structure["arrays"]["pos"] * ang2bohr
latticeP  = structure["cell"] * ang2bohr
lattice   = latticeP'
nat       = structure["N_atoms"]
species   = structure["arrays"]["species"]
println("Read structure with $nat atoms from $filename")

family    = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
atom_psps = [ElementPsp(Symbol(species[i]), family) for i in 1:nat]

invlat       = inv(lattice)
positionvecs = [invlat * positions[:, i] for i in 1:nat]

model = model_DFT(lattice, atom_psps, positionvecs;
                  spin_polarization=:none, functionals=PBE(), temperature=0.01)

if use_gpu
    import CUDA: CuArray
    arch  = DFTK.GPU(CuArray)
    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid, architecture=arch)
else
    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid)
end

kpt = basis.kpoints[1]
println("n_planewaves: ", length(G_vectors(basis, kpt)))
println("n_electrons:  ", basis.model.n_electrons)
println("ecut:         ", ecut)
println("kgrid:        ", kgrid)
println("use_gpu:      ", use_gpu)

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

println("\n" * "*"^80)
println("Precompilation\n")
self_consistent_field(basis; maxiter=2)
self_consistent_field(basis; maxiter=2, eigensolver=make_eigensolver(use_randomization=false))
self_consistent_field(basis; maxiter=2, eigensolver=make_eigensolver(use_randomization=true))
use_gpu && (GC.gc(); CUDA.reclaim())

println("\n" * "*"^80)
println("LOBPCG\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
scfres = self_consistent_field(basis)
println(DFTK.timer)
println(RandESC.timer)

println("\n" * "*"^80)
println("JD\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
scfres = self_consistent_field(basis; eigensolver=make_eigensolver(use_randomization=false))
println(DFTK.timer)
println(RandESC.timer)

println("\n" * "*"^80)
println("rand-JD\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
scfres = self_consistent_field(basis; eigensolver=make_eigensolver(use_randomization=true))
println(DFTK.timer)
println(RandESC.timer)
