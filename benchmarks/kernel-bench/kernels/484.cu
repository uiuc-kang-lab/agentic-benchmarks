#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void coalescedWarpMultiplyKernel(const float* __restrict__ A,
                                           float* __restrict__ C,
                                           float s,
                                           int64_t size)
{
    // Calculate aligned indices for this thread within its warp
    const int warpSize = 32;
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    const int globalWarpIdx = blockIdx.x * warpsPerBlock + warpIdx;
    
    // Each thread processes 4 elements, each warp processes 128 elements (32 threads * 4 elements)
    const int elementsPerWarp = warpSize * 4;
    int baseIdx = globalWarpIdx * elementsPerWarp + laneIdx * 4;
    
    // Process main aligned chunks
    if (baseIdx + 3 < size) {
        // Use vectorized load for coalesced memory access
        float4 data = *reinterpret_cast<const float4*>(A + baseIdx);
        
        // Process the four elements
        data.x *= s;
        data.y *= s;
        data.z *= s;
        data.w *= s;
        
        // Coalesced store
        *reinterpret_cast<float4*>(C + baseIdx) = data;
    }
    // Handle remaining elements at the end of the array
    else if (baseIdx < size) {
        // Process remaining elements individually while maintaining coalescing within the warp
        #pragma unroll
        for (int i = 0; i < 4 && baseIdx + i < size; i++) {
            C[baseIdx + i] = A[baseIdx + i] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Ensure block size is multiple of warp size for optimal coalescing
    const int threadsPerBlock = 256; // Must be multiple of 32
    const int warpsPerBlock = threadsPerBlock / 32;
    const int elementsPerWarp = 32 * 4;
    const int blocks = (size + (warpsPerBlock * elementsPerWarp - 1)) / (warpsPerBlock * elementsPerWarp);
    
    coalescedWarpMultiplyKernel<<<blocks, threadsPerBlock>>>(A.data_ptr<float>(),
                                                            C.data_ptr<float>(),
                                                            s,
                                                            size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced warp-aligned matrix-scalar multiplication");
}