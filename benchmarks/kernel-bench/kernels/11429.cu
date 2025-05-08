#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes KL divergence and performs a uniform, branchless block reduction
__global__ void kldiv_uniform_red_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Each thread computes a partial sum using a grid-stride loop
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float lp = log_predictions[i];
        float t  = targets[i];
        sum += expf(lp) - t * lp;
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // Perform block-level reduction in shared memory using branchless conditional arithmetic
    // Instead of using if(tid < s), we multiply by a flag that is 1 when true and 0 when false
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        float add_val = sdata[tid + s];
        // Compute flag as 1 if tid < s, else 0
        unsigned int flag = (tid < s);
        sdata[tid] += add_val * (float)flag;
        __syncthreads();
    }

    // The first thread adds the block's result to the global output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    const int shared_mem = threads * sizeof(float);

    kldiv_uniform_red_kernel<<<blocks, threads, shared_mem>>>(
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
