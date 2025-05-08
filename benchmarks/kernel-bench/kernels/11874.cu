#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int VEC_SIZE=4>
__global__ void kldiv_vectorized_kernel(
    const float* __restrict__ log_preds,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int64_t num_elements) {
    
    const int64_t tid = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    const int64_t stride = gridDim.x * blockDim.x * VEC_SIZE;
    float4 sum = {0.0f};

    // Vectorized processing main loop
    #pragma unroll 2
    for (int64_t i = tid; i < num_elements - VEC_SIZE + 1; i += stride) {
        const float4 log_vec = *reinterpret_cast<const float4*>(&log_preds[i]);
        const float4 tar_vec = *reinterpret_cast<const float4*>(&targets[i]);

        sum.x += __expf(log_vec.x) - tar_vec.x * log_vec.x;
        sum.y += __expf(log_vec.y) - tar_vec.y * log_vec.y;
        sum.z += __expf(log_vec.z) - tar_vec.z * log_vec.z;
        sum.w += __expf(log_vec.w) - tar_vec.w * log_vec.w;
    }

    // Handle remaining elements
    float rem_sum = 0.0f;
    for (int j = tid + (num_elements & ~(VEC_SIZE-1)); j < num_elements; j++) {
        rem_sum += __expf(log_preds[j]) - targets[j] * log_preds[j];
    }

    float thread_sum = sum.x + sum.y + sum.z + sum.w + rem_sum;

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Block-level reduction
    __shared__ float shared_sums[128];
    if (threadIdx.x % warpSize == 0) {
        shared_sums[threadIdx.x / warpSize] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x / warpSize; ++i) {
            block_sum += shared_sums[i];
        }
        atomicAdd(output, block_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_preds,
    const torch::Tensor& targets) {
    
    const int64_t num_elements = log_preds.numel();
    auto output = torch::zeros({1}, log_preds.options());

    // Optimized block sizing for H100
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (num_elements + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    kldiv_vectorized_kernel<<<grid_size, BLOCK_SIZE>>>(
        log_preds.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );

    return output / static_cast<float>(num_elements);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL Divergence optimized with vectorized balanced loads");
}