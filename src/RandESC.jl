module RandESC

    using TimerOutputs

    """TimerOutput object storing RandESC internal timings."""
    const timer = TimerOutput()

    """
    Shortened version of `@timeit` writing to the RandESC timer.
    Mirrors the DFTK `@timing` convention.
    """
    macro timing(args...)
        blocks = TimerOutputs.timer_expr(__source__, __module__, false,
                                        :(RandESC.timer), args...)
        if blocks isa Expr
            blocks  # function definition: timer_expr_func returns a single Expr
        else
            Expr(:block,
                blocks[1],
                Expr(:tryfinally,
                    :($(esc(args[end]))),
                    :($(blocks[2]))
                )
            )
        end
    end

    # list all functions that should be accessible from outside
    export randESCSolver, jdsym_rand_block, jdsym_block
    export timer, reset_timer!

    # include all source files from this module
    include("sketch.jl")
    include("randomization_utils.jl")
    include("jd_rand_block.jl")
    include("jd_block.jl")

    include("interface.jl")

    # ----------------------------- Utilities -----------------------------------

    function infer_v0(v0, n)
        if v0 !== nothing
            v = vec(v0);
        else
            v = randn(n)
        end
        v ./= max(norm(v), eps(real(eltype(v))))
        return v
    end

    # Calculate the norms of the columns of an array
    function columnwise_norms(X::AbstractArray)
        vec(sqrt.(sum(abs2, X; dims=1)))
    end

    # Calculate the dot poroducts of the columns of two arrays
    function columnwise_dots(A::AbstractArray, B::AbstractArray)
        vec(sum(conj(A) .* B; dims=1))
    end

    function sort_which(theta::AbstractVector, which::AbstractString)
        W = uppercase(which)
        if W == "LM"
            return sortperm(abs.(theta), rev=true)
        elseif W == "SM"
            return sortperm(abs.(theta), rev=false)
        elseif W == "LR"
            return sortperm(real.(theta), rev=true)
        elseif W == "SR"
            return sortperm(real.(theta), rev=false)
        elseif W == "LI"
            return sortperm(abs.(imag.(theta)), rev=true)
        elseif W == "SI"
            return sortperm(abs.(imag.(theta)), rev=false)
        else
            error("Unknown which = $which")
        end
    end

end