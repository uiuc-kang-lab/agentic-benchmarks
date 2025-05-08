#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// Store the scalar multiplier in constant memory
__constant__ float d_scalar;

// Kernel using vectorized loads/stores (float4) for improved memory coalescing.
__global__ void multiplyVectorizedKernel(const float4* __restrict__ A,
                                          float4* __restrict__ C,
                                          int64_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float s = d_scalar;
        float4 a = A[idx];
        a.x *= s;
        a.y *= s;
        a.z *= s;
        a.w *= s;
        C[idx] = a;
    }
}

// Kernel to handle any leftover elements if the total number is not a multiple of 4.
__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         int64_t start,
                                         int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int element = start + idx;
    if (element < size) {
        C[element] = A[element] * d_scalar;
    }
}

// Fallback simple kernel for unaligned data
__global__ void multiplySimpleKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * d_scalar;
    }
}

// Forward function: copies the scalar to constant memory, then dispatches the appropriate kernel
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;

    // Copy the scalar multiplier to constant memory
    cudaMemcpyToSymbol(d_scalar, &s, sizeof(float));

    // Check if both input and output are aligned for vectorized (float4) access
    bool aligned = ((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) % sizeof(float4)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(C.data_ptr<float>()) % sizeof(float4)) == 0);
    
    if (aligned && size >= 4) {
        int64_t count = size / 4;          // Number of float4 elements
        int remainder = size % 4;            // Leftover elements
        int blocks = (count + threads - 1) / threads;
        multiplyVectorizedKernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            count);
        
        // Process remaining elements
        if (remainder > 0) {
            int start = count * 4;
            int remBlocks = (remainder + threads - 1) / threads;
            multiplyRemainderKernel<<<remBlocks, threads>>>(
                A.data_ptr<float>(), C.data_ptr<float>(), start, size);
        }
    } else {
        // Fallback to simple kernel if data is not aligned
        int blocks = (size + threads - 1) / threads;
        multiplySimpleKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), size);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix-scalar multiplication kernel with constant scalar in constant memory and vectorized loads");
}
