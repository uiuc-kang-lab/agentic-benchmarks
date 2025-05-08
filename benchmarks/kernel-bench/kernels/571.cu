#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__constant__ float scalar;

__global__ void multiplyConstantMemoryKernel(const float4* __restrict__ A,
                                              float4* __restrict__ C,
                                              int64_t count) {
    // Calculate global index for this thread's warp
    const unsigned int warpSize = 32;
    const unsigned int warpId = threadIdx.x / warpSize;
    const unsigned int laneId = threadIdx.x % warpSize;
    const unsigned int warpsPerBlock = blockDim.x / warpSize;
    const unsigned int globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
    
    // Each thread processes 4 elements (float4)
    const unsigned int elementsPerWarp = warpSize * 4;
    const unsigned int startIdx = globalWarpIdx * elementsPerWarp / 4 + laneId;
    
    if (startIdx < count) {
        // Load and process data
        float4 data = A[startIdx];
        
        // Multiply using registers
        data.x *= scalar;
        data.y *= scalar;
        data.z *= scalar;
        data.w *= scalar;
        
        // Store result
        C[startIdx] = data;
    }
    // __syncwarp(); // Removed unnecessary warp synchronization
}

__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       int64_t start,
                                       int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (start + idx < size) {
        C[start + idx] = A[start + idx] * scalar;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Copy scalar to constant memory
    cudaMemcpyToSymbol(scalar, &s, sizeof(float));
    
    // Use 256 threads per block (8 warps)
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / 32;
    
    // Calculate number of float4 elements
    int64_t float4Count = size / 4;
    int remainder = size % 4;
    
    // Calculate grid size based on warp-level processing
    const int elementsPerWarp = 32 * 4; // each thread in warp processes float4
    const int warps = (float4Count + (elementsPerWarp/4) - 1) / (elementsPerWarp/4);
    const int blocks = (warps + warpsPerBlock - 1) / warpsPerBlock;

    multiplyConstantMemoryKernel<<<blocks, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        float4Count);

    if (remainder > 0) {
        int startIdx = float4Count * 4;
        int remainderBlocks = (remainder + threadsPerBlock - 1) / threadsPerBlock;
        multiplyRemainderKernel<<<remainderBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            C.data_ptr<float>(),
            startIdx,
            size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory optimized matrix-scalar multiplication kernel");
}