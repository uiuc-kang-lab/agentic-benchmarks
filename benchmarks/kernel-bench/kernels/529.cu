#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// This kernel combines vectorized memory accesses via float4 with loop unrolling in a grid-stride loop
// to improve memory throughput and reduce loop overhead.

__global__ void multiplyKernelCombined(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t size) {
    // Total number of threads
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process most of the data in groups of 4 floats using vectorized memory accesses
    int vectorCount = size / 4;  // Number of groups of 4 elements

    // Use grid-stride loop and unroll 4 iterations per loop to reduce loop overhead
    for (int i = tid; i < vectorCount; i += stride * 4) {
        #pragma unroll
        for (int offset = 0; offset < 4; offset++) {
            int idx = i + offset * stride;
            if (idx < vectorCount) {
                // Load 4 floats at once using __ldg for read-only caching
                float4 a = __ldg(reinterpret_cast<const float4*>(A) + idx);
                a.x *= s;
                a.y *= s;
                a.z *= s;
                a.w *= s;
                reinterpret_cast<float4*>(C)[idx] = a;
            }
        }
    }

    // Process the tail elements that are not a multiple of 4
    int tailStart = vectorCount * 4;
    for (int i = tailStart + tid; i < size; i += stride) {
        C[i] = A[i] * s;
    }
}

// Torch binding: wraps the kernel in a forward function

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();

    // Compute the number of vectorized (float4) elements
    int vectorCount = size / 4;
    const int threads = 256;
    // Use the vectorized portion to determine grid configuration
    int blocks = (vectorCount + threads - 1) / threads;

    multiplyKernelCombined<<<blocks, threads>>>(A.data_ptr<float>(),
                                                  C.data_ptr<float>(),
                                                  s,
                                                  size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient combined matrix-scalar multiplication kernel");
}
