#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel using shared memory and warp-level reduce pattern
__global__ void sharedMemoryReduceMultiplyKernel(const float* __restrict__ A,
                                                   float* __restrict__ C,
                                                   float s,
                                                   int64_t size) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    int gridSize = blockDim.x * 2 * gridDim.x;

    float mySum = 0;

    // Load data into shared memory, reducing over full warp width
    while (idx < size) {
        mySum += A[idx] * s + A[idx + blockDim.x] * s;
        idx += gridSize;
    }

    shared_data[tid] = mySum;
    __syncthreads();

    // Reduce within blocks using a single warp
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        C[blockIdx.x] = shared_data[0];
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads * 2 - 1) / (threads * 2);
    
    auto C = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    sharedMemoryReduceMultiplyKernel<<<blocks, threads, threads * sizeof(float)>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    
    // Final reduction on the CPU side
    return C.sum();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory reduction-based matrix-scalar multiplication kernel");
}