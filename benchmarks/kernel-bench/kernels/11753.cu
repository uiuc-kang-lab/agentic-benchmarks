#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 2D indexing kernel for KL divergence reduction

// Kernel with 2D thread block mapping (e.g., dim3(32, 8) for 256 threads per block)
__global__ void kl_div_kernel_2d(const float* __restrict__ log_predictions,
                                   const float* __restrict__ targets,
                                   float* __restrict__ output,
                                   const int n) {
    // Flattened thread index within the block
    const int block_size = blockDim.x * blockDim.y;
    const int local_id = threadIdx.x + threadIdx.y * blockDim.x;
    // Global index for grid-stride loop
    const int global_start = blockIdx.x * block_size + local_id;

    // Vectorized processing: each float4 loads 4 elements
    const int n4 = n / 4; // Number of float4 elements
    float sum = 0.0f;
    
    // Cast pointers to float4 for vectorized loads
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);
    
    // Process vectorized elements using grid-stride loop
    for (int idx = global_start; idx < n4; idx += gridDim.x * block_size) {
        float4 lp = __ldg(&logp_vec[idx]);
        float4 tt = __ldg(&targ_vec[idx]);
        sum += expf(lp.x) - tt.x * lp.x;
        sum += expf(lp.y) - tt.y * lp.y;
        sum += expf(lp.z) - tt.z * lp.z;
        sum += expf(lp.w) - tt.w * lp.w;
    }

    // Process remaining scalar elements if n is not multiple of 4
    const int scalar_start = n4 * 4;
    for (int idx = scalar_start + global_start; idx < n; idx += gridDim.x * block_size) {
        float lp = __ldg(log_predictions + idx);
        float tt = __ldg(targets + idx);
        sum += expf(lp) - tt * lp;
    }

    // Perform warp-level reduction using shfl_down_sync
    int lane = local_id & 31;  // equivalent to local_id % 32
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for storing warp sums; one per warp
    extern __shared__ float warp_sums[];
    int warp_id = local_id / 32;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the warp sums
    if (local_id < block_size / 32) {
        float block_sum = warp_sums[local_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (local_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA forward function exposed to PyTorch

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Use a 2D block: 32 threads in x and 8 threads in y --> 256 threads per block
    dim3 threads(32, 8);
    const int block_size = threads.x * threads.y;
    
    // Each float4 processes 4 elements, so determine number of blocks to cover n elements
    int blocks = (n + block_size * 4 - 1) / (block_size * 4);
    
    // Shared memory: one float per warp
    const int shared_mem = (block_size / 32) * sizeof(float);

    kl_div_kernel_2d<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Synchronize to ensure kernel completion (optional error checking can be added)
    cudaDeviceSynchronize();
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA 2D Indexing)");
}
