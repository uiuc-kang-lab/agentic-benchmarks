#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void balancedMultiplyKernel(const float* __restrict__ A,
                                        float* __restrict__ C,
                                        float s,
                                        int64_t size)
{
    extern __shared__ float sharedA[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Load data into shared memory
    if (idx < size) {
        sharedA[threadIdx.x] = __ldg(&A[idx]);
        __syncthreads();

        // Compute multiplication
        C[idx] = sharedA[threadIdx.x] * s;
    }

    // Parallel loop with balanced work allocation
    for (int i = idx + stride; i < size; i += stride) {
        C[i] = __ldg(&A[i]) * s;
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

    balancedMultiplyKernel<<<blocks, threads, threads * sizeof(float)>>>(A.data_ptr<float>(),
                                                    C.data_ptr<float>(),
                                                    s,
                                                    size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced workload matrix-scalar multiplication kernel");
}
