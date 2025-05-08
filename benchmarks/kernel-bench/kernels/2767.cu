#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void warp_leaky_relu_kernel(const float4* __restrict__ input,
                                     float4* __restrict__ output,
                                     const float negative_slope,
                                     const int n_vec) {
    // Each thread processes multiple float4 elements
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp_count = blockDim.x / WARP_SIZE;
    const unsigned int grid_stride = gridDim.x * blockDim.x / WARP_SIZE;
    
    // Shared memory for warp-level operations
    __shared__ float4 shared_data[32][33]; // +1 to avoid bank conflicts
    
    // Grid-stride loop over elements
    for (int idx = blockIdx.x * warp_count + wid; idx < n_vec; idx += grid_stride) {
        // Load data using vectorized access
        if (lane < 4) { // Only first 4 lanes per warp load data
            float4 val = input[idx * 8 + lane];
            
            // Process the float4 using warp-level parallelism
            val.x = val.x > 0.0f ? val.x : val.x * negative_slope;
            val.y = val.y > 0.0f ? val.y : val.y * negative_slope;
            val.z = val.z > 0.0f ? val.z : val.z * negative_slope;
            val.w = val.w > 0.0f ? val.w : val.w * negative_slope;
            
            // Store to shared memory
            shared_data[wid][lane] = val;
        }
        
        __syncwarp();
        
        // Write back results using coalesced stores
        if (lane < 4) {
            output[idx * 8 + lane] = shared_data[wid][lane];
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int n = x.numel();
    const int vec_size = sizeof(float4) / sizeof(float);
    const int n_vec = (n + vec_size - 1) / vec_size;
    
    // Configure launch parameters
    const int threads_per_block = 128; // 4 warps per block
    const int max_blocks = 1024;
    const int blocks = std::min(max_blocks, (n_vec + threads_per_block - 1) / threads_per_block);
    
    warp_leaky_relu_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        negative_slope,
        n_vec
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with warp optimizations");
}