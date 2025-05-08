#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float float4_t __attribute__((ext_vector_type(4)));

__global__ void kl_div_coalesced_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    // Vectorized index computation
    const int vec_size = 4;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vectors = (n + vec_size - 1) / vec_size;

    extern __shared__ float partial_sums[];
    float thread_sum = 0.0f;

    // Vectorized grid-stride loop with coalesced accesses
    for (int vec_idx = global_tid; vec_idx < num_vectors; vec_idx += gridDim.x * blockDim.x) {
        int scalar_idx = vec_idx * vec_size;
        int valid_size = min(vec_size, n - scalar_idx);

        // Vectorized load (coalesced)
        float4_t log_pred, target;
        *((float4_t*)&log_pred) = *reinterpret_cast<const float4_t*>(log_predictions + scalar_idx);
        *((float4_t*)&target) = *reinterpret_cast<const float4_t*>(targets + scalar_idx);

        // Process 4 elements (auto-unroll for efficiency)
        for (int i = 0; i < valid_size; ++i) {
            thread_sum += expf(log_pred[i]) - target[i] * log_pred[i];
        }
    }

    // Block reduction
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_vectorized_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Use 4x fewer threads to match vectorization
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;  // Account for vec_size

    kl_div_coalesced_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_vectorized_forward, "KLDiv with vectorized mem coalescing (CUDA)");
}