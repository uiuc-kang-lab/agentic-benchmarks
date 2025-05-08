#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void sharedMemoryMultiplyKernel(const float* __restrict__ A,
                                            float* __restrict__ C,
                                            float s,
                                            int64_t size)
{
    extern __shared__ float sharedA[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        sharedA[localIdx] = __ldg(&A[idx]);
    }
    __syncthreads();

    // Perform computation using shared memory
    if (idx < size) {
        C[idx] = sharedA[localIdx] * s;
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
    const size_t sharedMemSize = threads * sizeof(float);

    sharedMemoryMultiplyKernel<<<blocks, threads, sharedMemSize>>>(A.data_ptr<float>(),
                                                                   C.data_ptr<float>(),
                                                                   s,
                                                                   size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory optimized matrix-scalar multiplication kernel");
}