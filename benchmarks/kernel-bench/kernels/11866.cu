#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE = 256, int VECTOR_SIZE = 4>
__global__ void optimized_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * VECTOR_SIZE;
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    
    using float4_v = float4;
    float4_v sum = {0.0f, 0.0f, 0.0f, 0.0f};
    
    #pragma unroll
    for (int i = tid * VECTOR_SIZE; i < n; i += stride) {
        if (i + VECTOR_SIZE - 1 < n) {
            float4_v log_pred = *reinterpret_cast<const float4_v*>(&log_predictions[i]);
            float4_v target = *reinterpret_cast<const float4_v*>(&targets[i]);
            
            sum.x += __expf(log_pred.x) - target.x * log_pred.x;
            sum.y += __expf(log_pred.y) - target.y * log_pred.y;
            sum.z += __expf(log_pred.z) - target.z * log_pred.z;
            sum.w += __expf(log_pred.w) - target.w * log_pred.w;
        }
    }

    float thread_sum = sum.x + sum.y + sum.z + sum.w;

    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float warp_results[32];
    
    if (lane_id == 0) {
        warp_results[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (BLOCK_SIZE / warpSize)) {
        float warp_sum = warp_results[lane_id];
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    constexpr int BLOCK_SIZE = 256;
    constexpr int VECTOR_SIZE = 4;
    const int blocks = (n + VECTOR_SIZE * BLOCK_SIZE - 1) / (VECTOR_SIZE * BLOCK_SIZE);

    optimized_kldiv_kernel<BLOCK_SIZE, VECTOR_SIZE><<<blocks, BLOCK_SIZE>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized Vectorized KL Divergence Forward");
}