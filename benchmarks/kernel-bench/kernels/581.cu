#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// Kernel using vectorized loads (float4) without any unnecessary synchronizations
__global__ void multiplyVectorKernel(const float4* __restrict__ A,
                                       float4* __restrict__ C,
                                       float s,
                                       int64_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float4 a = A[idx];
        a.x *= s;
        a.y *= s;
        a.z *= s;
        a.w *= s;
        C[idx] = a;
    }
}

// Kernel to handle remaining elements if the total number is not divisible by 4
__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int64_t start,
                                          int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = start + idx;
    if (offset < size) {
        C[offset] = A[offset] * s;
    }
}

// Simple fallback kernel when data is not properly aligned
__global__ void multiplySimpleKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

// Forward function chooses the best kernel based on the alignment of the data
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const const int threads = 512;

    // Check if both input and output pointers are 16-byte aligned for float4 accesses
    bool aligned = ((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) % sizeof(float4)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(C.data_ptr<float>()) % sizeof(float4)) == 0);

    if (aligned && size >= 4) {
        int64_t count = size / 4; // Number of float4 elements
        int remainder = size % 4;
        int blocks = (count + threads - 1) / threads;
        multiplyVectorKernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            count);

        // Process any remaining elements
        if (remainder > 0) {
            int start = count * 4;
            int remBlocks = (remainder + threads - 1) / threads;
            multiplyRemainderKernel<<<remBlocks, threads>>>(
                A.data_ptr<float>(), C.data_ptr<float>(), s, start, size);
        }
    } else {
        int blocks = (size + threads - 1) / threads;
        multiplySimpleKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized matrix-scalar multiplication kernel with minimal synchronization");
}
