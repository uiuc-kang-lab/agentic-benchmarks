#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// The optimized kernel using a grid-stride loop for balanced workload
__global__ void gridStrideKernel(const float* __restrict__ A, float* __restrict__ C, float s, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements spaced by grid stride
    for (int i = idx; i < size; i += stride) {
        // Use read-only data cache for efficiency
        float a_val = __ldg(&A[i]);
        C[i] = a_val * s;
    }
}

// The forward function for PyTorch extension
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    // Dynamically compute grid size, ensuring full coverage of array
    const int blocks = (size + threads - 1) / threads;

    gridStrideKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride kernel for scalar multiplication");
}
