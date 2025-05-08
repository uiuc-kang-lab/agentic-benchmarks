#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel to compute KL divergence on a chunk of data
// Processes subarray starting at 'offset' with 'n' elements, and writes its partial result
// to partial_out[out_idx]. Assumes offset is multiple of 4 for vectorized loads (or accepts minor remainder).
__global__ void kl_div_kernel_overlap(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_out,
    const int n,         // number of elements in this chunk
    const int offset,    // starting index in the full array
    const int out_idx    // index in partial_out to accumulate the result
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_sums[];
    float sum = 0.0f;
    
    // Set pointers for the subarray
    const float* log_preds_sub = log_predictions + offset;
    const float* targets_sub   = targets + offset;
    
    int n4 = n / 4; // number of complete float4 elements
    const float4* logp_vec = reinterpret_cast<const float4*>(log_preds_sub);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets_sub);
    
    int vec_idx = global_idx;
    while (vec_idx < n4) {
        float4 logp = __ldg(&logp_vec[vec_idx]);
        float4 targ = __ldg(&targ_vec[vec_idx]);
        sum += expf(logp.x) - targ.x * logp.x
             + expf(logp.y) - targ.y * logp.y
             + expf(logp.z) - targ.z * logp.z
             + expf(logp.w) - targ.w * logp.w;
        vec_idx += gridDim.x * blockDim.x;
    }

    int scalar_start = n4 * 4; 
    int scalar_idx = scalar_start + global_idx;
    while (scalar_idx < n) {
        float lp = __ldg(log_preds_sub + scalar_idx);
        float tt = __ldg(targets_sub + scalar_idx);
        sum += expf(lp) - tt * lp;
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Warp-level reduction using shuffle
    for (int offset_sync = 16; offset_sync > 0; offset_sync >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset_sync);
    }
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces all warp sums
    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset_sync = 16; offset_sync > 0; offset_sync >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset_sync);
        }
        if (lane == 0) {
            atomicAdd(&partial_out[out_idx], block_sum);
        }
    }
}


// Forward function that splits the work into multiple streams and overlaps kernel execution with memory transfers

torch::Tensor kl_div_cuda_overlap_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    // Total number of elements
    const int n = log_predictions.numel();

    // Number of streams to use for overlapping computation and memory transfers
    const int num_streams = 4;
    int chunk_size = (n + num_streams - 1) / num_streams;

    // Allocate device tensor to hold partial results for each chunk
    auto options = log_predictions.options();
    torch::Tensor partial_out = torch::zeros({num_streams}, options);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);

    // Launch kernels in different streams, each processing a chunk
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        if (offset >= n) break;
        int current_chunk = std::min(chunk_size, n - offset);
        int blocks = (current_chunk + threads * 4 - 1) / (threads * 4);
        
        kl_div_kernel_overlap<<<blocks, threads, shared_mem, streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            partial_out.data_ptr<float>(),
            current_chunk,
            offset,
            i
        );
    }

    // Allocate pinned host memory for asynchronous copy of partial results
    float* host_partial = nullptr;
    cudaHostAlloc(&host_partial, num_streams * sizeof(float), cudaHostAllocDefault);

    // Asynchronously copy each stream's partial result from device to host
    for (int i = 0; i < num_streams; i++) {
        cudaMemcpyAsync(host_partial + i,
                        partial_out.data_ptr<float>() + i,
                        sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }

    // Synchronize all streams to ensure kernel execution and memory copies are complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Aggregate partial results from host pinned memory
    float total_sum = 0.0f;
    for (int i = 0; i < num_streams; i++) {
        total_sum += host_partial[i];
    }
    
    cudaFreeHost(host_partial);
    
    // Compute final result (average over all elements)
    float final_result = total_sum / static_cast<float>(n);
    
    // Return result as a tensor
    auto output = torch::full({1}, final_result, options);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_overlap_forward, "KL divergence forward with stream overlap (CUDA)");
}
