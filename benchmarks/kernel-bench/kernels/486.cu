#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Store the scalar multiplier in constant memory to benefit from the read-only cache
__constant__ float d_scalar;

// Kernel: vectorized multiplication using float4 and grid-stride loop
// The scalar is fetched from constant memory to reduce memory latency
__global__ void constantMemoryVectorizedMultiplyKernel(const float* __restrict__ A,
                                                          float* __restrict__ C,
                                                          int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    // Process complete groups of 4 elements
    int fullGroups = n / 4;
    for (int i = tid; i < fullGroups; i += totalThreads) {
        // reinterpret as float4 for vectorized load/store
        float4 a_val = reinterpret_cast<const float4*>(A)[i];
        float4 result;
        result.x = a_val.x * d_scalar;
        result.y = a_val.y * d_scalar;
        result.z = a_val.z * d_scalar;
        result.w = a_val.w * d_scalar;
        reinterpret_cast<float4*>(C)[i] = result;
    }
    
    // Process any remaining elements that do not form a full float4
    int offset = fullGroups * 4;
    for (int i = tid; i < (n - offset); i += totalThreads) {
        C[offset + i] = A[offset + i] * d_scalar;
    }
}

// Forward function that performs the input checks, copies the scalar to constant memory
// and launches the kernel
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();
    
    // Copy the scalar value to constant memory (read-only for all threads)
    cudaMemcpyToSymbol(d_scalar, &s, sizeof(float));
    
    const int threads = 256;
    const int blocks = (((n / 4) + threads - 1) / threads);
    
    constantMemoryVectorizedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                                 C.data_ptr<float>(),
                                                                 n);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication using constant memory and vectorized accesses");
}
