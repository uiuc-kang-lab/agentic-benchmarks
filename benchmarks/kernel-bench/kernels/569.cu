#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void multiplyBalancedKernel(const float4* __restrict__ A,
                                      float4* __restrict__ C,
                                      float s,
                                      int64_t count,
                                      int elementsPerThread) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements in a strided pattern
    for (int i = tid; i < count; i += stride) {
        float4 data = A[i];
        #pragma unroll
        {
            data.x *= s;
            data.y *= s;
            data.z *= s;
            data.w *= s;
        }
        C[i] = data;
    }
}

__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t start,
                                       int64_t size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (start + tid < size) {
        C[start + tid] = A[start + tid] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Optimal thread block size for H100
    const int threadsPerBlock = 256;
    
    // Calculate number of float4 elements
    int64_t float4Count = size / 4;
    int remainder = size % 4;
    
    // Target elements per thread for balanced workload
    const int targetElementsPerThread = 4;
    
    // Calculate optimal number of blocks
    int numSMs = 132; // H100 has 132 SMs
    int blocksPerSM = 32; // Targeting good occupancy
    int maxBlocks = numSMs * blocksPerSM;
    
    // Calculate actual number of blocks needed
    int blocks = min(maxBlocks, 
                    (int)((float4Count + threadsPerBlock * targetElementsPerThread - 1) 
                    / (threadsPerBlock * targetElementsPerThread)));

    multiplyBalancedKernel<<<blocks, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        s,
        float4Count,
        targetElementsPerThread);

    if (remainder > 0) {
        int startIdx = float4Count * 4;
        int remainderBlocks = (remainder + threadsPerBlock - 1) / threadsPerBlock;
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
    m.def("forward", &forward, "Balanced workload matrix-scalar multiplication kernel");
}