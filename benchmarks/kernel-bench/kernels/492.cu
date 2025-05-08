#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// New CUDA kernel that combines warp-level coalesced memory access,
// vectorized loads/stores using float4, and the __ldg intrinsic for improved read-only caching.

__global__ void efficientWarpVectorizedKernel(const float* __restrict__ A,
                                               float* __restrict__ C,
                                               float s,
                                               int64_t size) {
    // Define warp properties
    const int warpSize = 32;
    int warpIdx = threadIdx.x / warpSize;
    int laneIdx = threadIdx.x % warpSize;
    int warpsPerBlock = blockDim.x / warpSize;
    int globalWarpIdx = blockIdx.x * warpsPerBlock + warpIdx;

    // Each thread processes 4 contiguous elements, so each warp handles 32*4 = 128 elements
    const int elementsPerWarp = warpSize * 4;  // 128 elements per warp
    int baseIdx = globalWarpIdx * elementsPerWarp + laneIdx * 4;

    // Process main aligned chunk: load 4 floats using __ldg for better caching, then store with vectorized store
    if (baseIdx + 3 < size) {
        float4 a4;
        a4.x = __ldg(&A[baseIdx]);
        a4.y = __ldg(&A[baseIdx + 1]);
        a4.z = __ldg(&A[baseIdx + 2]);
        a4.w = __ldg(&A[baseIdx + 3]);

        // Multiply each component by s
        a4.x *= s;
        a4.y *= s;
        a4.z *= s;
        a4.w *= s;

        // Coalesced store
        *reinterpret_cast<float4*>(&C[baseIdx]) = a4;
    } else if (baseIdx < size) {
        // Handle any remaining elements individually
        #pragma unroll
        for (int i = 0; i < 4 && (baseIdx + i) < size; i++) {
            float val = __ldg(&A[baseIdx + i]);
            C[baseIdx + i] = val * s;
        }
    }
}

// Host function that prepares tensor and launches the kernel

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();

    // Set threads per block to a multiple of warp size
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / 32;
    const int elementsPerWarp = 32 * 4; // 128 elements per warp

    // Compute the number of blocks to cover the entire array
    int blocks = (size + (warpsPerBlock * elementsPerWarp) - 1) / (warpsPerBlock * elementsPerWarp);

    efficientWarpVectorizedKernel<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        size
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient warp-level vectorized and __ldg accelerated matrix-scalar multiplication");
}
