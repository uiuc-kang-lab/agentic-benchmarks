#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void hinge_loss_optimized_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int n
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // Each thread processes multiple elements (grid-stride loop)
    float thread_sum = 0.0f;
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        thread_sum += fmaxf(0.0f, 1.0f - predictions[i] * targets[i]);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32) {
        warpReduce<256>(sdata, tid);
    }

    // Write result for this block
    if (tid == 0) {
        block_results[bid] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    const int threads = 256;
    const int max_blocks = 1024;
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    
    auto options = predictions.options();
    auto block_results = torch::empty({blocks}, options);
    
    hinge_loss_optimized_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );

    // Compute final mean on CPU (more efficient for small number of blocks)
    auto sum = torch::sum(block_results);
    return sum.div(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Hinge Loss Forward");
}