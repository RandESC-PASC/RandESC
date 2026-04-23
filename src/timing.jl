using TimerOutputs
import TimerOutputs: reset_timer!

"""TimerOutput object storing RandESC internal timings."""
const timer = TimerOutput()

"""
Hooks called around every `@timing` block. Set by extensions (e.g. RandESCNVTXExt).
`timing_start_hook[](label)` is called before the block and its return value
(e.g. an NVTX RangeId) is forwarded to `timing_end_hook[]` after the block.
"""
const timing_start_hook = Ref{Function}(_ -> nothing)
const timing_end_hook   = Ref{Function}(_ -> nothing)

"""
Shortened version of `@timeit` writing to the RandESC timer.
Mirrors the DFTK `@timing` convention. Also fires `timing_start_hook` /
`timing_end_hook` so that extensions (e.g. NVTX) can inject profiling ranges.
"""
macro timing(args...)
    blocks = TimerOutputs.timer_expr(__source__, __module__, false,
                                    :(RandESC.timer), args...)
    if blocks isa Expr
        # Function definition: blocks = esc(combinedef(def)) = Expr(:escape, func_def).
        # Inject NVTX hooks around the timer-wrapped body inside the function.
        label = if length(args) == 2
            args[1]  # @timing "name" function f() ... end
        else         # @timing function f() ... end  →  derive from function name
            sig = args[1].args[1]
            string((sig isa Expr && sig.head === :where ? sig.args[1] : sig).args[1])
        end
        nvtx_id = gensym("nvtx_id")
        f = blocks.args[1]  # unwrap :escape
        f.args[end] = Expr(:block,
            :($nvtx_id = RandESC.timing_start_hook[]($label)),
            Expr(:tryfinally, f.args[end], :(RandESC.timing_end_hook[]($nvtx_id))))
        blocks
    else
        label   = args[1]
        nvtx_id = gensym("nvtx_id")
        Expr(:block,
            blocks[1],
            :($nvtx_id = RandESC.timing_start_hook[]($(esc(label)))),
            Expr(:tryfinally,
                :($(esc(args[end]))),
                Expr(:block,
                    :(RandESC.timing_end_hook[]($nvtx_id)),
                    blocks[2]
                )
            )
        )
    end
end
