using AMDGPU

include("symmetric_hermitian_random_matrix.jl")

if AMDGPU.has_rocm_gpu()
    run_symmetric_hermitian_random_matrix_tests(; template=AMDGPU.zeros(1))
else
    @warn "AMD GPU not detected; skipping AMD tests."
end