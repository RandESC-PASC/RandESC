using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using CUDA
using JSON3
using Dates
# Loading NVTX triggers RandESCNVTXExt, which injects NVTX ranges around every
# @timing block in RandESC — enabling timeline visualization in Nsight Systems.
using NVTX

# ── CLI ───────────────────────────────────────────────────────────────────────

function parse_args(args)
    opts = Dict{String,Any}(
        "structure"  => nothing,
        "solver"     => "jd",
        "ecut"       => 30.0,
        "kgrid"      => [2, 2, 2],
        "maxiter"    => 100,
        "tol"        => 1e-6,
        "scf_maxiter" => 10,  # keep low to avoid huge nsys dumps
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--solver";      opts["solver"]      = args[i+1]; i += 2
        elseif a == "--ecut";    opts["ecut"]        = parse(Float64, args[i+1]); i += 2
        elseif a == "--kgrid";   opts["kgrid"]       = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--maxiter"; opts["maxiter"]     = parse(Int, args[i+1]); i += 2
        elseif a == "--tol";     opts["tol"]         = parse(Float64, args[i+1]); i += 2
        elseif a == "--scf-maxiter"; opts["scf_maxiter"] = parse(Int, args[i+1]); i += 2
        elseif !startswith(a, "--") && isnothing(opts["structure"])
            opts["structure"] = a; i += 1
        else
            error("Unknown argument: $a")
        end
    end
    isnothing(opts["structure"]) && error("Usage: run_gpu_profile.jl <structure.extxyz> [options]")
    opts["solver"] in ("jd", "jd_sketched") ||
        error("GPU profiling only supports jd and jd_sketched (not lobpcg)")
    return opts
end

# ── Structure loading ─────────────────────────────────────────────────────────

function load_model(filename)
    structure    = read_frame(filename)
    ang2bohr     = 1.8897261249935897
    positions    = structure["arrays"]["pos"] * ang2bohr
    lattice      = structure["cell"]' * ang2bohr
    nat          = structure["N_atoms"]
    species      = structure["arrays"]["species"]
    family       = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
    atom_psps    = [ElementPsp(Symbol(species[i]), family) for i in 1:nat]
    positionvecs = [inv(lattice) * positions[:, i] for i in 1:nat]
    println("Loaded $nat-atom structure from $filename")
    return model_DFT(lattice, atom_psps, positionvecs;
                     spin_polarization=:none, functionals=PBE(), temperature=0.01)
end

# ── Eigensolver wrapper ───────────────────────────────────────────────────────

function make_eig_solver(use_randomization)
    function eig_solver(A, X0; prec=nothing, maxiter, tol, kwargs...)
        function precond_prep(M, X)
            DFTK.precondprep!(M, X)
        end
        return randESCSolver(A, X0, size(X0, 1), size(X0, 2);
                             preconditioner=prec,
                             precond_preparator=precond_prep,
                             maxiter=maxiter, tol=tol,
                             useRandomization=use_randomization,
                             verbose=false)
    end
    return eig_solver
end

# ── Main ──────────────────────────────────────────────────────────────────────

function run_gpu_profile(opts)
    structure_file = opts["structure"]
    solver         = opts["solver"]
    ecut           = opts["ecut"]
    kgrid          = opts["kgrid"]
    tol            = opts["tol"]
    scf_maxiter    = opts["scf_maxiter"]

    DFTK.setup_threading()
    println("CUDA device: $(CUDA.name(CUDA.device()))")

    arch  = DFTK.GPU(CuArray)
    model = load_model(structure_file)
    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid, architecture=arch)

    use_randomization = solver == "jd_sketched"
    eig_solver = make_eig_solver(use_randomization)

    # Warmup: compile all GPU kernels before profiling
    println("Warming up...")
    self_consistent_field(basis; maxiter=3, tol=1e-1, eigensolver=eig_solver)
    println("Warmup done.\n")

    system_name = splitext(basename(structure_file))[1]
    meta_file   = "$(system_name)_$(solver)_gpu_meta.json"

    # Profiling run — data written to report1.nsys-rep
    # nsys sends SIGTERM immediately after the capture range ends, so all
    # metadata writing must happen inside the @profile block before it closes.
    println("Profiling solver: $solver  (scf_maxiter=$scf_maxiter)")
    CUDA.@profile external=true begin
        scfres = self_consistent_field(basis; maxiter=scf_maxiter, tol=tol, eigensolver=eig_solver)

        report_candidates = filter(f -> occursin(r"^report\d*\.(nsys-rep|qdstrm)$", f), readdir("."))
        report_file = isempty(report_candidates) ? "report1.nsys-rep" :
                      sort(report_candidates, by=f -> mtime(f))[end]

        meta = Dict{String,Any}(
            "solver"        => solver,
            "system"        => system_name,
            "ecut"          => ecut,
            "kgrid"         => kgrid,
            "n_scf_iter"    => scfres.n_iter,
            "converged"     => scfres.converged,
            "n_matvec"      => scfres.n_matvec,
            "architecture"  => "gpu",
            "cuda_device"   => CUDA.name(CUDA.device()),
            "timestamp"     => string(now()),
            "julia_version" => string(VERSION),
            "report_file"   => report_file,
        )
        open(meta_file, "w") do io; JSON3.pretty(io, meta); end
        println("Metadata written to $meta_file")
        println("Profile data written to $report_file")
        if endswith(report_file, ".qdstrm")
            println("Note: nsys importer not available on this host. Copy $report_file")
            println("      to a machine with full nsys before running nsys_to_json.py.")
        end
        println("\nTo produce the JSON timing report run:")
        println("  python analysis/nsys_to_json.py --report $report_file --meta $meta_file --output results/$(system_name)_$(solver)_gpu.json")
    end
end

run_gpu_profile(parse_args(ARGS))
