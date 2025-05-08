#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

__global__ void block_reduce_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    const int vec_count = n / vec_size;
    
    // Shared memory for block reduction
    extern __shared__ float shared[];
    
    float thread_sum = 0.0f;

    // Vectorized processing
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 log_vec = __ldg(reinterpret_cast<const float4*>(log_predictions) + vec_idx);
        const float4 tgt_vec = __ldg(reinterpret_cast<const float4*>(targets) + vec_idx);
        
        thread_sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        thread_sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        thread_sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        thread_sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }

    // Handle remaining elements
    const int remaining_start = vec_count * vec_size;
    for (int i = remaining_start + tid; i < n; i += stride) {
        const float log_pred = __ldg(log_predictions + i);
        const float target = __ldg(targets + i);
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction first
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Store warp results in shared memory
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    if (lane_id == 0) {
        shared[warp_id] = thread_sum;
    }
    
    __syncthreads();

    // Complete block-level reduction using first warp
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        float warp_sum = shared[threadIdx.x];
        
        // Warp-level reduction of the block results
        #pragma unroll
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        // Single atomic add per block
        if (threadIdx.x == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor block_reduce_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch configuration
    const int threads = 256;
    const int blocks = std::min(256, (n + threads - 1) / threads);
    const int warps_per_block = threads / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);

    block_reduce_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_reduce_kl_forward, "Block-reduced KL divergence (CUDA)");
}