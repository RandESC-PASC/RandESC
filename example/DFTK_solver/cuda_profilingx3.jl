using DFTK
using PseudoPotentialData
using RandESC
using CUDA
using NVTX # this triggers the RandESCNVTXExt extension, which places NVTX markers for
           # nsight profiling. Avoid using in production, because of possible overhead

"""
This example shows how to profile RandESC solvers on NVIDIA GPUs using NVTX ranges. The
NVTX ranges are automatically injected around every `@timing` block in RandESC, thanks to
the RandESCNVTXExt extension. To get a profile, locate the `nsys` executable on your system
(usually located in the bin of the CUDA toolkit), then run:

    nsys launch julia --project=MyProject cuda_profiling.jl

Note that the first time DFTK is run on the GPU, precompilation can be quite lengthy.The data 
will be dumped to a file named `report1.nsys-rep`. To view this data in a timeline, you need
to move the file to your local machine, or generally be able to access it from your machine
(e.g. via sshfs). Provided you have installed a recent version of NVIDIA Nsigt systems, you
can then open the file with

    nsys-ui report1.nsys-rep

To get a statsitical summary instead of a timeline, you can run

    nsys stats --report nvtx_sum report1.nsys-rep

Final note: both the CUDA and NVTX packages are required for this example, but they are not
included in the local project environment (too heavy). You will need to add them on your own.
"""

randomize = false          # whether we profile randomized or standard JD
DFTK.setup_threading()    # Sets up default DFTK thread count
arch = DFTK.GPU(CuArray)  # Triggers DFTK to run on NVIDIA GPUs

# SrVO3 perovskite unit cell (cubic, five atoms per cell)
# a = 7.260                                                                                            
# lattice = a * [[1. 0. 0.];                                                                           
#                [0. 1. 0.];                                                                           
#                [0. 0. 1.]]                                                                           
#                                                                                                      
# Sr = ElementPsp(:Sr, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))                            
# V = ElementPsp(:V, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))                              
# O = ElementPsp(:O, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))                              
#                                                                                                      
# atoms     = [Sr, V, O, O, O]                                                                         
# positions = [[0.5, 0.5, 0.5],                                                                        
#              [0.0, 0.0, 0.0],                                                                        
#              [0.5, 0.0, 0.0],                                                                        
#              [0.0, 0.5, 0.0],                                                                        
#              [0.0, 0.0, 0.5]]                                                                        

a = 7.260
lattice = 3a * [[1. 0. 0.];
                [0. 1. 0.];
                [0. 0. 1.]]

Sr = ElementPsp(:Sr, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))
V  = ElementPsp(:V,  PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))
O  = ElementPsp(:O,  PseudoFamily("dojo.nc.sr.pbe.v0_4_1.stringent.upf"))

atoms = [
    fill(Sr, 27)...,
    fill(V,  27)...,
    fill(O,  27)...,
    fill(O,  27)...,
    fill(O,  27)...,
]

positions = vcat(
    # Sr at [0.5, 0.5, 0.5] in primitive cell -> [1/6, 1/2, 5/6] in supercell
    [[i/3 + 1/6, j/3 + 1/6, k/3 + 1/6] for i in 0:2 for j in 0:2 for k in 0:2],
    # V at [0.0, 0.0, 0.0]
    [[i/3 + 0.0, j/3 + 0.0, k/3 + 0.0] for i in 0:2 for j in 0:2 for k in 0:2],
    # O at [0.5, 0.0, 0.0]
    [[i/3 + 1/6, j/3 + 0.0, k/3 + 0.0] for i in 0:2 for j in 0:2 for k in 0:2],
    # O at [0.0, 0.5, 0.0]
    [[i/3 + 0.0, j/3 + 1/6, k/3 + 0.0] for i in 0:2 for j in 0:2 for k in 0:2],
    # O at [0.0, 0.0, 0.5]
    [[i/3 + 0.0, j/3 + 0.0, k/3 + 1/6] for i in 0:2 for j in 0:2 for k in 0:2],
)
                                                                                                     
model = model_DFT(lattice, atoms, positions; temperature=0.01, functionals=PBE())
basis = PlaneWaveBasis(model; Ecut=40, kgrid=[1, 1, 1], architecture=arch)

kpt = basis.kpoints[1]
println("n_planewaves: ", length(G_vectors(basis, kpt)))
println("n_electrons: ", basis.model.n_electrons)

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

# Warmup run: we do not want to profile compilation time
scfres = self_consistent_field(basis; maxiter=3, eigensolver=make_eigensolver(; use_randomization=randomize))
scfres = self_consistent_field(basis; maxiter=3)

maxit = 7

CUDA.@profile external=true begin
    self_consistent_field(basis; maxiter=maxit, eigensolver=make_eigensolver(; use_randomization=randomize))
end

t1 = time()
scfres = self_consistent_field(basis; maxiter=maxit)
t2 = time()
elapsed = t2 - t1
println("Elapsed lobpcg time $elapsed")
