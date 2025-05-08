#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void warpAlignedMultiplyKernel(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t size)
{
    // Calculate aligned size (multiple of 128 elements - 4 elements per thread * 32 threads per warp)
    const int aligned_size = (size / 128) * 128;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Process aligned portion with vectorized loads/stores
    int base_idx = global_warp_id * 128 + lane_id * 4;
    
    // Main processing loop for aligned data
    while (base_idx < aligned_size) {
        float4 data = *reinterpret_cast<const float4*>(A + base_idx);
        data.x *= s;
        data.y *= s;
        data.z *= s;
        data.w *= s;
        *reinterpret_cast<float4*>(C + base_idx) = data;
        
        base_idx += warps_per_block * gridDim.x * 128;
    }
    
    // Handle remainder with the first warp of the first block
    if (blockIdx.x == 0 && warp_id == 0) {
        base_idx = aligned_size + lane_id;
        while (base_idx < size) {
            C[base_idx] = A[base_idx] * s;
            base_idx += 32;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Use multiple of 32 threads for warp alignment
    const int threads = 256; // 8 warps per block
    const int blocks = (size + (threads * 4) - 1) / (threads * 4);
    
    warpAlignedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                  C.data_ptr<float>(),
                                                  s,
                                                  size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned matrix-scalar multiplication kernel");
}