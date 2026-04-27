using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using CUDA
using NVTX

function parse_args(args)
    opts = Dict{String,Any}(
        "structure"   => nothing,
        "solver"      => "jd",
        "ecut"        => 30.0,
        "kgrid"       => [2, 2, 2],
        "tol"         => 1e-6,
        "scf_maxiter" => 10,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--solver";          opts["solver"]      = args[i+1]; i += 2
        elseif a == "--ecut";        opts["ecut"]        = parse(Float64, args[i+1]); i += 2
        elseif a == "--kgrid";       opts["kgrid"]       = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--tol";         opts["tol"]         = parse(Float64, args[i+1]); i += 2
        elseif a == "--scf-maxiter"; opts["scf_maxiter"] = parse(Int, args[i+1]); i += 2
        elseif !startswith(a, "--") && isnothing(opts["structure"])
            opts["structure"] = a; i += 1
        else
            error("Unknown argument: $a")
        end
    end
    isnothing(opts["structure"]) && error("Usage: run_gpu_profile.jl <structure.extxyz> [options]")
    opts["solver"] in ("lobpcg", "jd", "jd_sketched") || error("solver must be lobpcg, jd, or jd_sketched")
    return opts
end

opts   = parse_args(ARGS)
solver = opts["solver"]

DFTK.setup_threading()
println("CUDA device: $(CUDA.name(CUDA.device()))")

structure = read_frame(opts["structure"])
ang2bohr  = 1.8897261249935897
lattice   = structure["cell"]' * ang2bohr
species   = structure["arrays"]["species"]
nat       = structure["N_atoms"]
family    = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
atoms     = [ElementPsp(Symbol(species[i]), family) for i in 1:nat]
positions = [inv(lattice) * (structure["arrays"]["pos"] * ang2bohr)[:, i] for i in 1:nat]
println("Loaded $nat-atom structure from $(opts["structure"])")

model = model_DFT(lattice, atoms, positions; spin_polarization=:none, functionals=PBE(), temperature=0.01)
basis = PlaneWaveBasis(model; Ecut=opts["ecut"], kgrid=opts["kgrid"], architecture=DFTK.GPU(CuArray))

function make_eigensolver(; useRandomization)
    function eigensolver(A, X0; prec=nothing, maxiter, tol, kwargs...)
        return randESCSolver(A, X0, size(X0, 1), size(X0, 2);
                             preconditioner=prec,
                             precond_preparator=(M, X) -> DFTK.precondprep!(M, X),
                             maxiter=maxiter, tol=tol,
                             useRandomization=useRandomization,
                             verbose=false)
    end
end

eigensolver_kwargs = solver == "lobpcg" ? (;) :
                     (eigensolver=make_eigensolver(; useRandomization=solver == "jd_sketched"),)

println("Warming up...")
self_consistent_field(basis; maxiter=3, tol=1e-1, eigensolver_kwargs...)
println("Warmup done.\n")

println("Profiling solver: $solver  (scf_maxiter=$(opts["scf_maxiter"]))")
CUDA.@profile external=true begin
    self_consistent_field(basis; maxiter=opts["scf_maxiter"], tol=opts["tol"], eigensolver_kwargs...)
end
