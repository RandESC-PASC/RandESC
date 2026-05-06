module RandESCNVTXExt
using NVTX
using RandESC
using CUDA

function RandESC.sync_nvtx_profiling(activate::Bool)
	if activate
	    @warn("Accurate NVTX profiling adds GPU synchronization overhead. "*
	          "Only use 'activate_nvtx_profiling()' if you are actively profiling.")
	    RandESC.timing_start_hook[] = label -> (CUDA.synchronize(); NVTX.range_start(; message=label, color=hash(label) % UInt32))
	    RandESC.timing_end_hook[] = token -> (CUDA.synchronize(); NVTX.range_end(token))
	end
end

end # module