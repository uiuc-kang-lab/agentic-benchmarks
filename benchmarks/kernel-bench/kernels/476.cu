#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void adaptiveMultiplyKernel(const float* __restrict__ A,
                                      float* __restrict__ C,
                                      float s,
                                      int64_t size,
                                      bool useSharedMem)
{
    extern __shared__ float sharedA[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if (useSharedMem) {
        // Shared memory path for larger data sizes
        if (idx < size) {
            sharedA[threadIdx.x] = __ldg(&A[idx]);
            __syncthreads();
            C[idx] = sharedA[threadIdx.x] * s;
        }

        for (int i = idx + stride; i < size; i += stride) {
            C[i] = __ldg(&A[i]) * s;
        }
    } else {
        // Direct memory path for smaller data sizes
        if (idx < size) {
            C[idx] = A[idx] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Threshold to decide whether to use shared memory
    const int64_t SHARED_MEM_THRESHOLD = 1024;
    bool useSharedMem = (size >= SHARED_MEM_THRESHOLD);
    
    adaptiveMultiplyKernel<<<blocks, threads, useSharedMem ? threads * sizeof(float) : 0>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        size,
        useSharedMem
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive matrix-scalar multiplication kernel");
}