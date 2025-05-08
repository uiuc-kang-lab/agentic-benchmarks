#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Device function to multiply a float4 by a scalar
__device__ __forceinline__ float4 multiply_float4(const float4 a, const float s) {
    float4 res;
    res.x = a.x * s;
    res.y = a.y * s;
    res.z = a.z * s;
    res.w = a.w * s;
    return res;
}

// Modular device function to process vectorized (float4) loads and stores
__device__ __forceinline__ void process_vectorized(const float* __restrict__ A, float* __restrict__ C, float s, int idx) {
    // Read one vector of 4 floats
    float4 a_val = ((const float4 *)A)[idx];
    float4 result = multiply_float4(a_val, s);
    ((float4 *)C)[idx] = result;
}

// Modular device function to process a single float element
__device__ __forceinline__ void process_scalar(const float* __restrict__ A, float* __restrict__ C, float s, int idx) {
    C[idx] = __ldg(&A[idx]) * s;
}

// Kernel using modular device functions and grid-stride loops
__global__ void modularMultiplyKernel(const float* __restrict__ A,
                                        float* __restrict__ C,
                                        float s,
                                        int64_t n) {
    // Number of full groups of 4 elements
    int fullGroups = n / 4;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process vectorized groups using a grid-stride loop
    for (int i = globalIdx; i < fullGroups; i += stride) {
         process_vectorized(A, C, s, i);
    }
    
    // Process any remaining elements with a single thread (since remainder < 4)
    if (globalIdx == 0) {
        for (int i = fullGroups * 4; i < n; i++) {
            process_scalar(A, C, s, i);
        }
    }
}

// The forward function that prepares the output tensor and launches the kernel
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    // Determine launch configuration based on vectorized groups
    const int threads = 256;
    const int blocks = ((n / 4) + threads - 1) / threads;

    modularMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 s,
                                                 n);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular device function based matrix-scalar multiplication kernel");
}
