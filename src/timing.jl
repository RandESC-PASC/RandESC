using TimerOutputs
import TimerOutputs: reset_timer!

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
