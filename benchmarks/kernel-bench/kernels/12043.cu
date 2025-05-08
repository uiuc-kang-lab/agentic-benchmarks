#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_fast_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ result,
    const int n
) {
    __shared__ float sdata[256];
    
    const int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Local accumulator
    float sum = 0.0f;
    
    // Process multiple elements per thread
    while (gid < n) {
        const float pred = __ldg(&predictions[gid]);
        const float target = __ldg(&targets[gid]);
        sum += fmaxf(0.0f, 1.0f - pred * target);
        gid += stride;
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block - unrolled for last 6 iterations
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    
    // Last warp reduction without synchronization
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    const int n = predictions.numel();
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    
    auto partial_sums = torch::empty({blocks}, predictions.options());
    
    hinge_loss_fast_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );
    
    // Final reduction on GPU
    return torch::sum(partial_sums) / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}