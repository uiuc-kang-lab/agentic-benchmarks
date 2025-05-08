#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define VECTOR_WIDTH 4
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Warp-level reduction using shuffle intrinsics for a single warp
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that uses vectorized loads (float4) to ensure memory coalescing
__global__ void kl_div_kernel_coalesced(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    int n_vec = n / VECTOR_WIDTH;
    int rem = n % VECTOR_WIDTH;
    
    // Cast pointers to float4 for vectorized (coalesced) access
    const float4* log_ptr = reinterpret_cast<const float4*>(log_predictions);
    const float4* tar_ptr = reinterpret_cast<const float4*>(targets);

    // Process bulk elements using vectorized loads
    for (int i = gid; i < n_vec; i += gridSize) {
        float4 lp = log_ptr[i];
        float4 tp = tar_ptr[i];
        thread_sum += expf(lp.x) - tp.x * lp.x;
        thread_sum += expf(lp.y) - tp.y * lp.y;
        thread_sum += expf(lp.z) - tp.z * lp.z;
        thread_sum += expf(lp.w) - tp.w * lp.w;
    }

    // Process any remaining elements
    int offset = n_vec * VECTOR_WIDTH;
    for (int i = offset + gid; i < n; i += gridSize) {
        float lp = log_predictions[i];
        float tp = targets[i];
        thread_sum += expf(lp) - tp * lp;
    }
    
    // Warp-level reduction using shuffle
    thread_sum = warp_reduce(thread_sum);

    // Shared memory reduction across warps within the block
    extern __shared__ float warp_sums[];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    float block_sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
    block_sum = warp_reduce(block_sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// PyTorch interface function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = BLOCK_SIZE;
    int total_threads = (n + VECTOR_WIDTH - 1) / VECTOR_WIDTH; // approximate work units based on vector width
    const int blocks = (total_threads + threads - 1) / threads;
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    kl_div_kernel_coalesced<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA coalesced)");
}
