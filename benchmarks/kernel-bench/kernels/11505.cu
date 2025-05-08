#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Ensure coalesced memory access by having consecutive threads
    // access consecutive memory locations within a warp
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Base index for this warp ensures coalesced access
    const int base_idx = global_warp_id * (warp_size * 4) + lane_id;
    
    extern __shared__ float shmem[];
    float sum = 0.0f;
    
    // Each thread processes 4 elements with stride of warp_size
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        const int idx = base_idx + i * warp_size;
        if(idx < n) {
            // Aligned vector loads for better memory throughput
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for(int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if(lane_id == 0) {
        partial_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if(threadIdx.x < warps_per_block) {
        float warp_sum = partial_sums[threadIdx.x];
        if(threadIdx.x == 0) {
            for(int i = 1; i < warps_per_block; i++) {
                warp_sum += partial_sums[i];
            }
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int elements_per_block = threads * 4;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}