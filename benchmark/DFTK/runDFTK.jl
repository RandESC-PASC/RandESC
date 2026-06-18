using ExtXYZ
using DFTK
using PseudoPotentialData
using RandESC
using Dates
using Printf

# Parse CLI arguments: [--gpu] [--ecut=N] [--kgrid=N,N,N] [filename]
use_gpu = "--gpu" in ARGS
get_flag(prefix) = let m = match(Regex("^$prefix=(.+)"), something(findfirst(a -> startswith(a, "$prefix="), ARGS) |> i -> i === nothing ? nothing : ARGS[i], ""))
    m === nothing ? nothing : m.captures[1]
end
ecut_arg       = get_flag("--ecut")
kgrid_arg      = get_flag("--kgrid")
results_dir_arg = get_flag("--results-dir")
ecut        = ecut_arg  === nothing ? 45 : parse(Int, ecut_arg)
kgrid       = kgrid_arg === nothing ? [2, 2, 2] : parse.(Int, split(kgrid_arg, ","))
results_dir = results_dir_arg === nothing ? "results" : results_dir_arg
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
                  spin_polarization=:none, functionals=PBE(), temperature=0.001)

if use_gpu
    import CUDA
    arch  = DFTK.GPU(CUDA.CuArray)
    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid, architecture=arch)
else
    basis = PlaneWaveBasis(model; Ecut=ecut, kgrid=kgrid)
end

kpt = basis.kpoints[1]
n_pw = length(G_vectors(basis, kpt))
n_el = basis.model.n_electrons
println("n_planewaves: ", n_pw)
println("n_electrons:  ", n_el)
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

# Extract time in seconds from a TimerOutput, returning NaN if the path doesn't exist
function get_timer_time(t, keys...)
    cur = t
    for k in keys
        haskey(cur.inner_timers, k) || return NaN
        cur = cur.inner_timers[k]
    end
    cur.accumulated_data.time / 1e9
end

const CSV_HEADER = "timestamp,structure,ecut,kgrid,n_planewaves,n_electrons,use_gpu,algorithm,total_scf,eigensolver_total,matvec,diag,ortho,allocation"

function save_timings(algorithm, dftk_t, randesc_t)
    mkpath(results_dir)
    csv_path = joinpath(results_dir, "timings.csv")
    if !isfile(csv_path) || readline(csv_path) != CSV_HEADER
        open(io -> println(io, CSV_HEADER), csv_path, "w")
    end

    total_scf = get_timer_time(dftk_t, "self_consistent_field")
    if algorithm == "lobpcg"
        eigensolver_total = get_timer_time(dftk_t, "self_consistent_field", "LOBPCG")
        matvec     = get_timer_time(dftk_t, "self_consistent_field", "LOBPCG", "DftHamiltonian multiplication")
        diag       = get_timer_time(dftk_t, "self_consistent_field", "LOBPCG", "rayleigh_ritz")
        ortho      = get_timer_time(dftk_t, "self_consistent_field", "LOBPCG", "ortho! X vs Y")
        allocation = NaN
    elseif algorithm == "jd"
        eigensolver_total = get_timer_time(randesc_t, "jd")
        matvec     = get_timer_time(randesc_t, "jd", "jd: matvec")
        diag       = get_timer_time(randesc_t, "jd", "jd: diag")
        ortho      = get_timer_time(randesc_t, "jd", "jd: ortho")
        allocation = get_timer_time(randesc_t, "jd", "jd: allocation")
    else
        eigensolver_total = get_timer_time(randesc_t, "jd_sketched")
        matvec     = get_timer_time(randesc_t, "jd_sketched", "jd_sketched: matvec")
        diag       = get_timer_time(randesc_t, "jd_sketched", "jd_sketched: diag")
        ortho      = get_timer_time(randesc_t, "jd_sketched", "jd_sketched: ortho")
        allocation = get_timer_time(randesc_t, "jd_sketched", "jd_sketched: allocation")
    end

    fmt(x)      = isnan(x) ? "" : @sprintf("%.4f", x)
    struct_name = splitext(basename(filename))[1]
    kgrid_str   = join(kgrid, "x")
    ts          = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    open(csv_path, "a") do io
        println(io, "$ts,$struct_name,$ecut,$kgrid_str,$n_pw,$n_el,$use_gpu,$algorithm,$(fmt(total_scf)),$(fmt(eigensolver_total)),$(fmt(matvec)),$(fmt(diag)),$(fmt(ortho)),$(fmt(allocation))")
    end
end

println("\n" * "*"^80)
println("Precompilation\n")
self_consistent_field(basis; maxiter=2)
self_consistent_field(basis; maxiter=2, eigensolver=make_eigensolver(use_randomization=false))
self_consistent_field(basis; maxiter=2, eigensolver=make_eigensolver(use_randomization=true))
use_gpu && (GC.gc(); CUDA.reclaim())
use_gpu && RandESC.activate_nvtx_profiling()
use_gpu && (DFTK.timing_sync[] = CUDA.synchronize)

println("\n" * "*"^80)
println("LOBPCG\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
use_gpu && (GC.gc(); CUDA.reclaim())
self_consistent_field(basis)
println(DFTK.timer)
println(RandESC.timer)
save_timings("lobpcg", DFTK.timer, RandESC.timer)

println("\n" * "*"^80)
println("JD\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
use_gpu && (GC.gc(); CUDA.reclaim())
self_consistent_field(basis; eigensolver=make_eigensolver(use_randomization=false))
println(DFTK.timer)
println(RandESC.timer)
save_timings("jd", DFTK.timer, RandESC.timer)

println("\n" * "*"^80)
println("rand-JD\n")
DFTK.reset_timer!(DFTK.timer)
RandESC.reset_timer!(RandESC.timer)
use_gpu && (GC.gc(); CUDA.reclaim())
self_consistent_field(basis; eigensolver=make_eigensolver(use_randomization=true))
println(DFTK.timer)
println(RandESC.timer)
save_timings("jd_sketched", DFTK.timer, RandESC.timer)
