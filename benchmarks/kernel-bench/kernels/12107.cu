#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int BLOCK_SIZE>
__global__ void hinge_loss_optimized_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ global_sum,
    const int n
) {
    __shared__ float shared_sum[BLOCK_SIZE + 32]; // Increased shared memory to avoid bank conflicts
    
    const int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    float thread_sum = 0.0f;
    
    while (idx < n) {
        thread_sum += fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
        
        if (idx + BLOCK_SIZE < n) {
            thread_sum += fmaxf(0.0f, 1.0f - predictions[idx + BLOCK_SIZE] * targets[idx + BLOCK_SIZE]);
        }
        
        idx += gridDim.x * blockDim.x * 2;
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    #pragma unroll
    for (int stride = BLOCK_SIZE/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* vmem = shared_sum;
        if (BLOCK_SIZE >= 64) vmem[tid] += vmem[tid + 32];
        if (BLOCK_SIZE >= 32) vmem[tid] += vmem[tid + 16];
        if (BLOCK_SIZE >= 16) vmem[tid] += vmem[tid + 8];
        if (BLOCK_SIZE >= 8)  vmem[tid] += vmem[tid + 4];
        if (BLOCK_SIZE >= 4)  vmem[tid] += vmem[tid + 2];
        if (BLOCK_SIZE >= 2)  vmem[tid] += vmem[tid + 1];
    }
    
    if (tid == 0) {
        atomicAdd(global_sum, shared_sum[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    const int n = predictions.numel();
    auto options = predictions.options();
    
    auto global_sum_tensor = torch::zeros({1}, options);
    float* global_sum_ptr = global_sum_tensor.data_ptr<float>();
    
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = min(256, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    hinge_loss_optimized_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum_ptr,
        n
    );
    
    float mean_loss = global_sum_tensor.item<float>() / n;
    return torch::full({}, mean_loss, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Hinge Loss Forward");
}