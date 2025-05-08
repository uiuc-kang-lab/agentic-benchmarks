#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float float4_t __attribute__((ext_vector_type(4)));

__global__ void vectorized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int vector_size = 4;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vector_size;
    
    float4_t log_pred_vec, target_vec;
    float sum = 0.0f;

    const float4_t* log_pred_ptr = reinterpret_cast<const float4_t*>(log_predictions);
    const float4_t* target_ptr = reinterpret_cast<const float4_t*>(targets);

    // Vectorized processing
    while (idx < n - vector_size + 1) {
        log_pred_vec = log_pred_ptr[(blockIdx.x * blockDim.x + threadIdx.x)];
        target_vec = target_ptr[(blockIdx.x * blockDim.x + threadIdx.x)];

        for (int i = 0; i < vector_size; ++i) {
            sum += expf(log_pred_vec[i]) - target_vec[i] * log_pred_vec[i];
        }
        
        idx += blockDim.x * gridDim.x * vector_size;
    }

    // Handle remaining elements
    idx = (blockIdx.x * blockDim.x + threadIdx.x) * vector_size;
    for (int i = 0; i < vector_size; ++i) {
        if (idx + i < n) {
            float log_pred = log_predictions[idx + i];
            float target = targets[idx + i];
            sum += expf(log_pred) - target * log_pred;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction
    __shared__ float shared_sums[32];
    if (threadIdx.x % 32 == 0) {
        shared_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) {
            block_sum += shared_sums[i];
        }
        atomicAdd(output, block_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    vectorized_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Vectorized KL divergence forward (CUDA)");
}