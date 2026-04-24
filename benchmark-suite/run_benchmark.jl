using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using JSON3
using Dates
using FFTW
using LinearAlgebra

# ── CLI ───────────────────────────────────────────────────────────────────────

function parse_args(args)
    opts = Dict{String,Any}(
        "structure" => nothing,
        "output"    => "results",
        "ecut"      => 30.0,
        "kgrid"     => [2, 2, 2],
        "solvers"      => ["lobpcg", "jd", "jd_sketched"],
        "maxiter"      => 100,
        "tol"          => 1e-6,
        "fftw_threads" => 1,
        "blas_threads" => 1,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--output";   opts["output"]       = args[i+1]; i += 2
        elseif a == "--ecut"; opts["ecut"]         = parse(Float64, args[i+1]); i += 2
        elseif a == "--kgrid"; opts["kgrid"]       = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--solvers"; opts["solvers"]   = split(args[i+1], ","); i += 2
        elseif a == "--maxiter"; opts["maxiter"]   = parse(Int, args[i+1]); i += 2
        elseif a == "--tol";  opts["tol"]          = parse(Float64, args[i+1]); i += 2
        elseif a == "--fftw-threads"; opts["fftw_threads"] = parse(Int, args[i+1]); i += 2
        elseif a == "--blas-threads"; opts["blas_threads"] = parse(Int, args[i+1]); i += 2
        elseif !startswith(a, "--") && isnothing(opts["structure"])
            opts["structure"] = a; i += 1
        else
            error("Unknown argument: $a")
        end
    end
    isnothing(opts["structure"]) && error("Usage: run_benchmark.jl <structure.extxyz> [options]")
    return opts
end

# ── Timer helpers ─────────────────────────────────────────────────────────────

function get_timer_ns(to, keys...)
    t = to
    for k in keys
        haskey(t.inner_timers, k) || return Int64(0)
        t = t.inner_timers[k]
    end
    return t.accumulated_data.time
end

ns_to_s(ns) = ns * 1e-9

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

# ── Timing extraction ─────────────────────────────────────────────────────────

function extract_timings(solver, dftk_t, randesc_t)
    scf = "self_consistent_field"
    hmul = "DftHamiltonian multiplication"

    if solver == "lobpcg"
        return (
            eigensolver_time_s = ns_to_s(get_timer_ns(dftk_t, scf, "LOBPCG")),
            matvec_time_s      = ns_to_s(get_timer_ns(dftk_t, scf, "LOBPCG", hmul)),
            ortho_time_s       = ns_to_s(get_timer_ns(dftk_t, scf, "LOBPCG", "ortho!") +
                                         get_timer_ns(dftk_t, scf, "LOBPCG", "ortho! X vs Y")),
            rr_time_s          = ns_to_s(get_timer_ns(dftk_t, scf, "LOBPCG", "rayleigh_ritz")),
            residual_time_s    = ns_to_s(get_timer_ns(dftk_t, scf, "LOBPCG", "Update residuals")),
        )
    else
        prefix = solver  # "jd" or "jd_sketched"
        # Sub-timers are nested under the outer "jd"/"jd_sketched" block
        return (
            eigensolver_time_s = ns_to_s(get_timer_ns(randesc_t, prefix)),
            matvec_time_s      = ns_to_s(get_timer_ns(randesc_t, prefix, "$prefix: matvec")),
            ortho_time_s       = ns_to_s(get_timer_ns(randesc_t, prefix, "$prefix: ortho")),
            rr_time_s          = ns_to_s(get_timer_ns(randesc_t, prefix, "$prefix: diag")),
            residual_time_s    = ns_to_s(get_timer_ns(randesc_t, prefix, "$prefix: residual")),
        )
    end
end

# ── Warmup ────────────────────────────────────────────────────────────────────

function warmup(model, solvers)
    println("Warming up (compiling all code paths)...")
    warmup_basis = PlaneWaveBasis(model; Ecut=4, kgrid=[1, 1, 1])
    self_consistent_field(warmup_basis; maxiter=2, tol=1e-1)
    if any(s in solvers for s in ("jd", "jd_sketched"))
        self_consistent_field(warmup_basis; maxiter=2, tol=1e-1,
                              eigensolver=make_eig_solver(false))
    end
    if "jd_sketched" in solvers
        self_consistent_field(warmup_basis; maxiter=2, tol=1e-1,
                              eigensolver=make_eig_solver(true))
    end
    println("Warmup done.\n")
end

# ── Main ──────────────────────────────────────────────────────────────────────

function run_benchmark(opts)
    structure_file = opts["structure"]
    output_dir     = opts["output"]
    ecut           = opts["ecut"]
    kgrid          = opts["kgrid"]
    solvers        = opts["solvers"]
    tol            = opts["tol"]
    fftw_threads   = opts["fftw_threads"]
    blas_threads   = opts["blas_threads"]
    system_name    = splitext(basename(structure_file))[1]

    FFTW.set_num_threads(fftw_threads)
    BLAS.set_num_threads(blas_threads)
    println("Threads — Julia: $(Threads.nthreads())  FFTW: $fftw_threads  BLAS: $blas_threads")

    mkpath(output_dir)
    model = load_model(structure_file)
    warmup(model, solvers)

    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid)

    for solver in solvers
        println("=" ^ 60)
        println("Running solver: $solver")

        DFTK.reset_timer!(DFTK.timer)
        RandESC.reset_timer!(RandESC.timer)

        scfres = if solver == "lobpcg"
            self_consistent_field(basis; tol=tol)
        elseif solver == "jd"
            self_consistent_field(basis; tol=tol,
                                  eigensolver=make_eig_solver(false))
        elseif solver == "jd_sketched"
            self_consistent_field(basis; tol=tol,
                                  eigensolver=make_eig_solver(true))
        else
            error("Unknown solver: $solver. Valid choices: lobpcg, jd, jd_sketched")
        end

        println(DFTK.timer)
        println(RandESC.timer)

        timings = extract_timings(solver, DFTK.timer, RandESC.timer)

        result = Dict{String,Any}(
            "solver"             => solver,
            "system"             => system_name,
            "ecut"               => ecut,
            "kgrid"              => kgrid,
            "n_scf_iter"         => scfres.n_iter,
            "converged"          => scfres.converged,
            "n_matvec"           => scfres.n_matvec,
            "wall_time_s"        => ns_to_s(get_timer_ns(DFTK.timer, "self_consistent_field")),
            "eigensolver_time_s" => timings.eigensolver_time_s,
            "matvec_time_s"      => timings.matvec_time_s,
            "ortho_time_s"       => timings.ortho_time_s,
            "rr_time_s"          => timings.rr_time_s,
            "residual_time_s"    => timings.residual_time_s,
            "timestamp"          => string(now()),
            "julia_version"      => string(VERSION),
            "julia_threads"      => Threads.nthreads(),
            "fftw_threads"       => fftw_threads,
            "blas_threads"       => blas_threads,
        )

        outfile = joinpath(output_dir, "$(system_name)_$(solver).json")
        open(outfile, "w") do io
            JSON3.pretty(io, result)
        end
        println("Written to $outfile\n")
    end
end

run_benchmark(parse_args(ARGS))
