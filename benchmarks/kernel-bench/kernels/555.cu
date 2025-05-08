#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// Kernel using vectorized loads (float4) with loop unrolling for improved performance.
__global__ void multiplyVectorizedKernelUnrolled(const float4* __restrict__ A,
                                                  float4* __restrict__ C,
                                                  float s,
                                                  int64_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float4 a = A[idx];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((float*)&a)[i] *= s;
        }
        C[idx] = a;
    }
}

// Kernel to handle any leftover elements if the total number is not divisible by 4.
__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int64_t start,
                                          int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int element = start + idx;
    if (element < size) {
        C[element] = A[element] * s;
    }
}

// Fallback simple kernel for unaligned or small arrays
__global__ void multiplySimpleKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

// The forward function checks alignment and utilizes vectorized loads for better memory coalescing.
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 32;

    // Check if both input and output pointers are 16-byte aligned for float4 accesses.
    bool aligned = ((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) % sizeof(float4)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(C.data_ptr<float>()) % sizeof(float4)) == 0);

    if (aligned && size >= 4) {
        int64_t count = size / 4;          // Number of float4 elements
        int remainder = size % 4;            // Elements remaining after vectorized processing
        int blocks = (count + threads - 1) / threads;
        multiplyVectorizedKernelUnrolled<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            count);

        // Process remainder elements if any
        if (remainder > 0) {
            int start = count * 4;
            int remBlocks = (remainder + threads - 1) / threads;
            multiplyRemainderKernel<<<remBlocks, threads>>>(
                A.data_ptr<float>(), C.data_ptr<float>(), s, start, size);
        }
    } else {
        // Fallback to the simple kernel if data isn't properly aligned
        int blocks = (size + threads - 1) / threads;
        multiplySimpleKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix-scalar multiplication kernel with vectorized loads and loop unrolling");
}
