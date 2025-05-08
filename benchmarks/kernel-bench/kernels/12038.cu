#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void hinge_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    float* __restrict__ partial_sums,
    const int n
) {
    __shared__ float shared_mem[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop to handle multiple elements per thread
    for (int idx = gid; idx < n; idx += grid_stride) {
        thread_sum += fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
    }
    
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    const int n = predictions.numel();
    const int blocks = min(256, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    auto partial_sums = torch::empty(blocks, predictions.options());
    
    hinge_loss_kernel<<<blocks, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        nullptr,  // output array not needed
        partial_sums.data_ptr<float>(),
        n
    );
    
    // Compute final mean on GPU
    return torch::sum(partial_sums) / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}