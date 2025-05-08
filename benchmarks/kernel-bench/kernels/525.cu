#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    __shared__ float shared_s;
    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();
    
    const unsigned int warp_size = 32;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int grid_stride = gridDim.x * blockDim.x;
    
    // Calculate base index for this thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread using vectorized loads
    while (idx * 4 + 3 < size) {
        float4 data = __ldg(reinterpret_cast<const float4*>(A + idx * 4));
        
        // Multiply using shared scalar
        data.x *= shared_s;
        data.y *= shared_s;
        data.z *= shared_s;
        data.w *= shared_s;
        
        // Store result
        reinterpret_cast<float4*>(C)[idx] = data;
        
        idx += grid_stride;
    }
    
    // Handle remaining elements
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * 4 + (size & ~3);  // Start from where vectorized processing ended
    
    while (idx < size) {
        C[idx] = A[idx] * shared_s;
        idx += grid_stride;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Use multiple of warp size for thread block
    const int threads = 256;  // 8 warps per block
    // Calculate optimal number of blocks based on SM count and occupancy
    const int max_blocks_per_sm = 2048 / threads;  // Assuming max 2048 threads per SM
    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int target_blocks_total = num_sms * max_blocks_per_sm;  // Target optimal occupancy
    
    // Ensure we have enough blocks to cover the data, but don't exceed optimal count
    const int min_blocks_needed = (size + (threads * 4 - 1)) / (threads * 4);
    const int blocks = std::min(target_blocks_total, min_blocks_needed);

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}