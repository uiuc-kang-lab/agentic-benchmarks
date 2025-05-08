#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_idx = tid * 4;
    __shared__ float partial_sums[256];
    
    float sum = 0.0f;
    
    // Process 4 consecutive elements with coalesced access
    for(int i = 0; i < 4; ++i) {
        const int idx = base_idx + i;
        if(idx < n) {
            const float log_pred = __ldg(log_predictions + idx);
            const float target = __ldg(targets + idx);
            sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Block reduction with optimized indexing
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Atomic add with warp-aggregated pattern
    if(threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;
    
    // Cap blocks to 144 SMs * 4 blocks/SM for H100
    const int max_blocks = 144 * 4;
    const int final_blocks = min(blocks, max_blocks);
    
    kl_div_kernel<<<final_blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward optimized (CUDA)");
}