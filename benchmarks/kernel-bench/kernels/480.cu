#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA kernel using grid-stride loops to handle workloads larger than available threads
__global__ void strideLoopMultiplyKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        C[i] = A[i] * s;
    }
}

// Forward function that checks inputs and launches the CUDA kernel
// It uses a stride loop to ensure each thread handles multiple elements if needed
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    // Launch enough threads so that each thread can process multiple elements via the stride loop
    const int blocks = (size + threads - 1) / threads;

    strideLoopMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                    C.data_ptr<float>(),
                                                    s,
                                                    size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop matrix-scalar multiplication kernel");
}
