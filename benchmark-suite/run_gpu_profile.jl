using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using CUDA
using JSON3
using Dates
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
    opts["solver"] in ("jd", "jd_sketched") || error("solver must be jd or jd_sketched")
    return opts
end

opts        = parse_args(ARGS)
solver      = opts["solver"]
scf_maxiter = opts["scf_maxiter"]

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

println("Warming up...")
self_consistent_field(basis; maxiter=3, tol=1e-1,
                      eigensolver=make_eigensolver(; useRandomization=solver == "jd_sketched"))
println("Warmup done.\n")

println("Profiling solver: $solver  (scf_maxiter=$scf_maxiter)")
CUDA.@profile external=true begin
    scfres = self_consistent_field(basis; maxiter=scf_maxiter, tol=opts["tol"],
                                   eigensolver=make_eigensolver(; useRandomization=solver == "jd_sketched"))
end

system_name = splitext(basename(opts["structure"]))[1]
meta_file   = "$(system_name)_$(solver)_gpu_meta.json"
open(meta_file, "w") do io
    JSON3.pretty(io, Dict{String,Any}(
        "solver"        => solver,
        "system"        => system_name,
        "ecut"          => opts["ecut"],
        "kgrid"         => opts["kgrid"],
        "n_scf_iter"    => scfres.n_iter,
        "converged"     => scfres.converged,
        "n_matvec"      => scfres.n_matvec,
        "architecture"  => "gpu",
        "cuda_device"   => CUDA.name(CUDA.device()),
        "timestamp"     => string(now()),
        "julia_version" => string(VERSION),
    ))
end
println("Metadata written to $meta_file")
