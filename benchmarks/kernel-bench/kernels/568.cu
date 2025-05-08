#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel that leverages shared memory to stage a tile of the input before processing
// Each block loads a tile of input elements into shared memory, multiplies by the scalar in shared memory, and writes back to global memory

__global__ void sharedMemMultiplyKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int n) {
    extern __shared__ float tile[];  // dynamically allocated shared memory
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load from global memory into shared memory
    if (globalIdx < n) {
        tile[localIdx] = A[globalIdx];
    }
    __syncthreads();

    // Perform multiplication in shared memory
    if (globalIdx < n) {
        tile[localIdx] *= s;
    }
    __syncthreads();

    // Write the result from shared memory back to global memory
    if (globalIdx < n) {
        C[globalIdx] = tile[localIdx];
    }
}

// Forward function to setup kernel execution

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int n = A.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Allocate shared memory: one float per thread in the block
    size_t sharedMemSize = threads * sizeof(float);

    sharedMemMultiplyKernel<<<blocks, threads, sharedMemSize>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        n
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel using shared memory");
}
