#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void multiplyWarpShuffleKernel(const float4* __restrict__ A,
                                         float4* __restrict__ C,
                                         float s,
                                         int64_t count) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warpId = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    const unsigned int globalIdx = blockIdx.x * blockDim.x + tid;
    
    // Broadcast scalar value within warp using shuffle
    float scalar = __shfl_sync(0xffffffff, s, 0);
    
    // Each thread processes multiple float4 elements
    int64_t stride = gridDim.x * blockDim.x;
    int64_t i = globalIdx;
    
    #pragma unroll 4
    for (; i + stride * 3 < count; i += stride * 4) {
        float4 data1 = A[i];
        float4 data2 = A[i + stride];
        float4 data3 = A[i + stride * 2];
        float4 data4 = A[i + stride * 3];
        
        data1.x *= scalar; data1.y *= scalar; data1.z *= scalar; data1.w *= scalar;
        data2.x *= scalar; data2.y *= scalar; data2.z *= scalar; data2.w *= scalar;
        data3.x *= scalar; data3.y *= scalar; data3.z *= scalar; data3.w *= scalar;
        data4.x *= scalar; data4.y *= scalar; data4.z *= scalar; data4.w *= scalar;
        
        C[i] = data1;
        C[i + stride] = data2;
        C[i + stride * 2] = data3;
        C[i + stride * 3] = data4;
    }
    
    // Handle remaining elements
    for (; i < count; i += stride) {
        float4 data = A[i];
        data.x *= scalar;
        data.y *= scalar;
        data.z *= scalar;
        data.w *= scalar;
        C[i] = data;
    }
}

__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t start,
                                       int64_t size) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int globalIdx = blockIdx.x * blockDim.x + tid;
    
    // Broadcast scalar value within warp using shuffle
    float scalar = __shfl_sync(0xffffffff, s, 0);
    
    if (start + globalIdx < size) {
        C[start + globalIdx] = A[start + globalIdx] * scalar;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Use 128 threads per block (4 warps) for better occupancy
    const int threadsPerBlock = 128;
    const int numSMs = 144; // H100 has 144 SMs
    const int blocksPerSM = 2; // Launch 2 blocks per SM for better occupancy
    const int maxBlocks = numSMs * blocksPerSM;
    
    // Calculate number of float4 elements
    int64_t float4Count = size / 4;
    int remainder = size % 4;
    
    // Launch kernel with enough blocks to saturate the GPU
    int blocks = min(maxBlocks, (int)((float4Count + threadsPerBlock - 1) / threadsPerBlock));
    
    multiplyWarpShuffleKernel<<<blocks, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        s,
        float4Count);

    if (remainder > 0) {
        int startIdx = float4Count * 4;
        int remainderBlocks = min(maxBlocks, (remainder + threadsPerBlock - 1) / threadsPerBlock);
        multiplyRemainderKernel<<<remainderBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            C.data_ptr<float>(),
            s,
            startIdx,
            size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-shuffle optimized matrix-scalar multiplication kernel");
}