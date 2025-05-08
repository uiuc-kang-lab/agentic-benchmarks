#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp/warpgroup size constants
constexpr int WARP_SIZE = 32;

__global__ void ldg_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    const int vec_count = n / vec_size;
    const int remaining = n % vec_size;

    float sum = 0.0f;

    // Vectorized processing - aligned 16B accesses
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 log_vec = __ldg(reinterpret_cast<const float4*>(log_predictions) + vec_idx);
        const float4 tgt_vec = __ldg(reinterpret_cast<const float4*>(targets) + vec_idx);
        
        sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }

    // Process remaining elements
    const int scalar_tid = vec_count * vec_size + tid;
    for (int i = scalar_tid; i < n; i += stride) {
        sum += expf(__ldg(log_predictions + i)) - __ldg(targets + i) * __ldg(log_predictions + i);
    }

    // Warp reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Shared memory for warp partial sums
    extern __shared__ float warp_sums[];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0)
        warp_sums[warp_id] = sum;
    
    __syncthreads();

    // Final reduction and atomics (single warp only)
    if (warp_id == 0 && lane_id < (blockDim.x / WARP_SIZE)) {
        float val = warp_sums[lane_id];
        for (int offset = WARP_SIZE/2; offset >= 1; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        
        if (lane_id == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor ldg_kl_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch config optimized for H100's 68 SMs
    const int threads = 256;
    const int blocks = 128;  // 4 warps per block x128 = 512 concurrent warps
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    ldg_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ldg_kl_cuda_forward, "KL divergence optimized with __ldg memory access (CUDA)");
}