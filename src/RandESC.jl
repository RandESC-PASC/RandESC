module RandESC

    # list all functions that should be accessible from outside
    export rand_ira, ira, lobpcg, sketched_to_fully, randESCSolver, lobpcg_softlock, lobpcg_lock, lobpcg_sketch_lock, jdsym, jdsym_rand

    # include all source files from this module
    include("sketch.jl")
    include("sketched_to_fully.jl")
    include("IRA.jl")
    include("RIRA.jl")
    include("LOBPCG.jl")
    include("LOBPCG_sketch.jl")
    include("jd.jl")
    include("jd_rand.jl")

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