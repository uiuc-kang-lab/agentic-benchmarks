#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Warp-level reduction using shuffle down
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_vectorized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    constexpr int VEC_SIZE = 4;  // using float4 for vectorized loads
    
    // Calculate global thread index and overall stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Determine how many full float4 loads we can do
    int vec_n = n / VEC_SIZE;
    float sum = 0.0f;
    
    // Cast global pointers to float4 for aligned accesses.
    const float4* log_preds_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targets_vec = reinterpret_cast<const float4*>(targets);
    
    // Process elements in chunks of 4 (vectorized load ensures coalescing)
    for (int i = tid; i < vec_n; i += stride) {
        float4 lp = log_preds_vec[i];
        float4 tg = targets_vec[i];
        sum += expf(lp.x) - tg.x * lp.x;
        sum += expf(lp.y) - tg.y * lp.y;
        sum += expf(lp.z) - tg.z * lp.z;
        sum += expf(lp.w) - tg.w * lp.w;
    }
    
    // Handle remaining elements (tail), if n is not a multiple of 4
    int tail_start = vec_n * VEC_SIZE;
    for (int i = tail_start + tid; i < n; i += stride) {
        float lp = log_predictions[i];
        float tg = targets[i];
        sum += expf(lp) - tg * lp;
    }
    
    // Perform warp-level reduction using shuffle operations
    sum = warp_reduce_sum(sum);

    // Shared memory reduction across warps in the block
    __shared__ float shared[32];  // allocate enough for up to 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();
    
    // First warp reduces the results from all warps
    if (warp_id == 0) {
        int num_warps = blockDim.x / warpSize;
        float block_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    constexpr int VEC_SIZE = 4;
    const int threads = 256;  // Must be a multiple of warpSize (32)
    const int blocks = std::min(256, (n + threads * VEC_SIZE - 1) / (threads * VEC_SIZE));
    const int shared_mem = (threads / 32) * sizeof(float);
    
    kl_div_vectorized_kernel<<<blocks, threads, shared_mem>>>(
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
