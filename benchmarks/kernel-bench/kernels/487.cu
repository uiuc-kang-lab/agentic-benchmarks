#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel that evenly distributes both vectorized and remainder computation across all threads
__global__ void evenWorkloadMultiplyKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process data in groups of 4 using vectorized loads/stores
    int fullGroups = n / 4;
    int remainder = n % 4;
    
    for (int i = tid; i < fullGroups; i += stride) {
        // Cast pointer to float4 for vectorized access
        float4 a_val = ((const float4*)A)[i];
        float4 result;
        result.x = a_val.x * s;
        result.y = a_val.y * s;
        result.z = a_val.z * s;
        result.w = a_val.w * s;
        ((float4*)C)[i] = result;
    }
    
    // Process any remaining elements with a grid-stride loop to evenly distribute the load
    int base = fullGroups * 4;
    for (int i = tid; i < remainder; i += stride) {
        C[base + i] = A[base + i] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    const int threads = 256;
    // Calculate blocks based on the number of full vector groups
    const int blocks = ((n / 4) + threads - 1) / threads;

    evenWorkloadMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                      C.data_ptr<float>(),
                                                      s,
                                                      n);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride even-workload vectorized matrix-scalar multiplication kernel");
}
