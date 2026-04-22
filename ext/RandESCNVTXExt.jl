module RandESCNVTXExt
using NVTX
using RandESC

function __init__()
	RandESC.timing_start_hook[] = label -> NVTX.range_start(; message=label, color=hash(label) % UInt32)
	RandESC.timing_end_hook[] = NVTX.range_end
end

end # module