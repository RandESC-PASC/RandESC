run_unit        = isempty(ARGS) || "unit"        in ARGS
run_integration = isempty(ARGS) || "integration" in ARGS

if run_unit
    println("Running unit tests...")
    include("unit/symmetric_hermitian_random_matrix.jl")
    include("unit/test_orthogonalization.jl")
end

if run_integration
    println("Running integration tests...")
    include("integration/dftk_silicon_test.jl")
end
