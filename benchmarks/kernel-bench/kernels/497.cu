#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Device function to multiply a float4 by a scalar
__device__ float4 multiply_float4(const float4 a, const float s) {
    float4 res;
    res.x = a.x * s;
    res.y = a.y * s;
    res.z = a.z * s;
    res.w = a.w * s;
    return res;
}

// Device function for vectorized processing
__device__ void process_vectorized(const float* __restrict__ A, float* __restrict__ C, float s, int idx) {
    float4 a_val = ((const float4 *)A)[idx];
    float4 result = multiply_float4(a_val, s);
    ((float4 *)C)[idx] = result;
}

// Device function for scalar processing
__device__ void process_scalar(const float* __restrict__ A, float* __restrict__ C, float s, int idx) {
    C[idx] = A[idx] * s;
}

// Kernel using modular device functions
__global__ void modularDeviceFunctionKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int fullGroups = size / 4;

    // Process vectorized groups
    for (int i = idx; i < fullGroups; i += stride) {
        process_vectorized(A, C, s, i);
    }

    // Process remaining elements
    for (int i = 4 * fullGroups + idx; i < size; i += stride) {
        process_scalar(A, C, s, i);
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    modularDeviceFunctionKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                     C.data_ptr<float>(),
                                                     s,
                                                     size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular device function optimized matrix-scalar multiplication kernel");
}
