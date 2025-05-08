#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int VEC_SIZE = 4;

__global__ void balanced_vectorized_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Vectorized processing (4 elements/thread)
    const int vec_count = n / VEC_SIZE;
    for (int i = tid; i < vec_count; i += total_threads) {
        float4 lp = __ldg(reinterpret_cast<const float4*>(log_predictions) + i);
        float4 t = __ldg(reinterpret_cast<const float4*>(targets) + i);
        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Process remaining elements with grid-stride loop
    const int residual_start = vec_count * VEC_SIZE;
    for (int i = residual_start + tid; i < n; i += total_threads) {
        float lp = __ldg(log_predictions + i);
        float t = __ldg(targets + i);
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block-level reduction
    if (threadIdx.x % WARP_SIZE == 0) {
        extern __shared__ float warp_sums[];
        warp_sums[threadIdx.x/WARP_SIZE] = sum;
        __syncthreads();

        if (threadIdx.x < WARP_SIZE) {
            float block_sum = (threadIdx.x < blockDim.x/WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            
            if (threadIdx.x == 0)
                atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor balanced_vectorized_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int vec_elements = n / VEC_SIZE;
    int blocks = (vec_elements + threads - 1) / threads;
    const int max_blocks = 512;
    blocks = min(blocks, max_blocks);
    const int shared_mem = (threads/WARP_SIZE) * sizeof(float);

    balanced_vectorized_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_vectorized_kl_forward, "Balanced vectorized KL divergence (CUDA)");
}