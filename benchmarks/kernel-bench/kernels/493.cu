#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Combined kernel: uses warp-level coalesced vectorized loads and a grid-stride loop over warps
// to maximize memory throughput and load balancing.
__global__ void combinedWarpHybridKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int64_t n) {
    const int warpSize = 32;
    // Each block is divided into warps
    int warpsPerBlock = blockDim.x / warpSize;
    int warpIdInBlock = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    // Total warps available in the grid
    int totalWarps = gridDim.x * warpsPerBlock;
    // Global warp index assigned to this thread's warp
    int globalWarpIdx = blockIdx.x * warpsPerBlock + warpIdInBlock;

    // Each warp processes 32 (warpSize) threads that each load 4 elements = 128 elements per warp
    const int elementsPerWarp = warpSize * 4;

    // Grid-stride loop over warps
    for (int warp = globalWarpIdx; warp * elementsPerWarp < n; warp += totalWarps) {
        // Each thread in the warp processes 4 consecutive elements
        int base = warp * elementsPerWarp + laneId * 4;

        // If we have a full float4 available, use vectorized load/store
        if (base + 3 < n) {
            float4 data = *reinterpret_cast<const float4*>(A + base);
            data.x *= s;
            data.y *= s;
            data.z *= s;
            data.w *= s;
            *reinterpret_cast<float4*>(C + base) = data;
        } 
        // Fallback for remaining elements that do not form a full float4
        else if (base < n) {
            #pragma unroll
            for (int i = 0; i < 4 && base + i < n; i++) {
                C[base + i] = A[base + i] * s;
            }
        }
    }
}

// Host forward function integrated with PyTorch
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    // Set thread configuration (must be a multiple of warp size)
    const int threadsPerBlock = 256;
    const int warpSize = 32;
    const int warpsPerBlock = threadsPerBlock / warpSize;
    const int elementsPerWarp = warpSize * 4;
    // Compute minimum number of blocks to cover the data in one iteration
    const int blocks = (n + (warpsPerBlock * elementsPerWarp - 1)) / (warpsPerBlock * elementsPerWarp);

    combinedWarpHybridKernel<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        n
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined warp-level and grid-stride matrix-scalar multiplication kernel");
}
