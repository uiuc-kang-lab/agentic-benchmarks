#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, 
                                const float* __restrict__ targets, 
                                float* __restrict__ output, 
                                const int n) {
    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and compute initial values
    float sum = 0.0f;
    while(idx < n) {
        sum += fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = sum;
    
    // Only synchronize before shared memory operations
    __syncthreads();

    // Reduction in shared memory
    for(int s = blockDim.x/2; s > 32; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within a warp)
    if(tid < 32) {
        volatile float* smem = sdata;
        if(blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if(blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if(blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if(blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if(blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if(blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    // Write result for this block to global mem
    if(tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    const int threads = 256;
    const int blocks = min(65535, (n + threads - 1) / threads);
    
    auto options = predictions.options();
    auto block_results = torch::empty({blocks}, options);
    
    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );

    auto sum = torch::sum(block_results);
    return sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}