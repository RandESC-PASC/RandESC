using CUDA

include("symmetric_hermitian_random_matrix.jl")
include("test_orthogonalization.jl")

if CUDA.has_cuda() && CUDA.has_cuda_gpu()
    run_symmetric_hermitian_random_matrix_tests(; template=CUDA.zeros(1))
    run_orthogonalization_tests(; template=CUDA.zeros(1))
else
    @warn "CUDA GPU not detected; skipping CUDA tests."
end