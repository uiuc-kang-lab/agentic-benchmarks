#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int VECTOR_SIZE = 4;

__global__ void vectorized_ldg_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector_stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Process 128-bit aligned vector loads
    const float4* pred_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);
    
    for (int i = tid; i < n/VECTOR_SIZE; i += vector_stride) {
        float4 lp = __ldg(&pred_vec[i]);
        float4 t = __ldg(&targ_vec[i]);
        
        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Process remaining elements with scalar loads
    const int remaining = n - (n/VECTOR_SIZE)*VECTOR_SIZE;
    for (int i = tid + (n/VECTOR_SIZE)*VECTOR_SIZE; i < n; i += vector_stride) {
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
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x/WARP_SIZE; i++)
            block_sum += warp_sums[i];
        atomicAdd(output, block_sum);
    }
}

torch::Tensor vectorized_ldg_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + VECTOR_SIZE*threads - 1) / (VECTOR_SIZE*threads), 512);
    const int shared_mem = (threads/WARP_SIZE) * sizeof(float);

    vectorized_ldg_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_ldg_kl_forward, "Vectorized LDG KL divergence (CUDA)");
}