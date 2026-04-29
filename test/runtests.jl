run_unit        = isempty(ARGS) || "unit"        in ARGS
run_integration = isempty(ARGS) || "integration" in ARGS
# Only run GPU tests if explicitly requested
run_amdgpu      =                  "amdgpu"      in ARGS
run_cuda        =                  "cuda"        in ARGS

if run_unit
    println("Running CPU unit tests...")
    include("unit/cpu.jl")
end

if run_integration
    println("Running integration tests...")
    include("integration/dftk_silicon_test.jl")
end

if run_amdgpu
    println("Running AMDGPU unit tests...")
    include("unit/amdgpu.jl")
end

if run_cuda
    println("Running CUDA unit tests...")
    include("unit/cuda.jl")
end
