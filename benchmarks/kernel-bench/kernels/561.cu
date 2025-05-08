#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// Kernel for processing vectorized data (float4) using a grid-stride loop with computed iterations.
// This minimizes warp divergence by avoiding per-iteration boundary checks.
__global__ void multiplyVectorizedKernelGridStride(const float4* __restrict__ A,
                                                    float4* __restrict__ C,
                                                    float s,
                                                    int64_t count) {
    // Total number of threads in the grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    // Each thread will process 'iterations' elements uniformly
    int iterations = count / totalThreads;
    int extra = count % totalThreads;  // first 'extra' threads process one additional element

    for (int i = 0; i < iterations; ++i) {
        int idx = tid + i * totalThreads;
        // Unrolled multiplication for vectorized load
        float4 a = A[idx];
        a.x *= s;
        a.y *= s;
        a.z *= s;
        a.w *= s;
        C[idx] = a;
    }
    // Process one extra element for threads with tid < extra
    if (tid < extra) {
        int idx = tid + iterations * totalThreads;
        float4 a = A[idx];
        a.x *= s;
        a.y *= s;
        a.z *= s;
        a.w *= s;
        C[idx] = a;
    }
}

// Kernel for processing remaining elements (non-multiple of 4) using a grid-stride loop.
__global__ void multiplyRemainderKernelGridStride(const float* __restrict__ A,
                                                   float* __restrict__ C,
                                                   float s,
                                                   int start,
                                                   int total) {
    int remainderCount = total - start;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    int iterations = remainderCount / totalThreads;
    int extra = remainderCount % totalThreads;

    for (int i = 0; i < iterations; ++i) {
        int idx = start + tid + i * totalThreads;
        float a = A[idx];
        C[idx] = a * s;
    }
    if (tid < extra) {
        int idx = start + tid + iterations * totalThreads;
        float a = A[idx];
        C[idx] = a * s;
    }
}

// Fallback kernel for cases when data isn't properly aligned for vectorized (float4) accesses.
// Uses a grid-stride loop to uniformly process elements.
__global__ void multiplySimpleKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int idx = tid; idx < total; idx += nthreads) {
        float a = A[idx];
        C[idx] = a * s;
    }
}

// Forward function: selects between the vectorized kernel and the simple kernel based on data alignment for float4
// The vectorized kernel uses a grid-stride loop with uniform control flow to minimize warp divergence.

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;

    // Check 16-byte alignment for float4 accesses
    bool aligned = ((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) % sizeof(float4)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(C.data_ptr<float>()) % sizeof(float4)) == 0);

    if (aligned && size >= 4) {
        // Process data in groups of 4 floats using vectorized load/store
        int64_t float4Count = size / 4;  // number of float4 elements
        int remainder = size % 4;         // leftover elements

        // Compute grid dimensions for the vectorized kernel
        int blocks = (float4Count + threads - 1) / threads;
        multiplyVectorizedKernelGridStride<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            float4Count
        );

        // Process any remaining elements
        if (remainder > 0) {
            int start = float4Count * 4;
            int remBlocks = (remainder + threads - 1) / threads;
            multiplyRemainderKernelGridStride<<<remBlocks, threads>>>(
                A.data_ptr<float>(),
                C.data_ptr<float>(),
                s,
                start,
                size
            );
        }
    } else {
        int blocks = (size + threads - 1) / threads;
        multiplySimpleKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication with uniform grid-stride loops to minimize warp divergence");
}
