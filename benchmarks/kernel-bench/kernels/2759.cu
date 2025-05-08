#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREAD_COUNT (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void warp_leaky_relu_kernel(const float4* __restrict__ input, 
                                     float4* __restrict__ output, 
                                     float* __restrict__ shared_mem,
                                     const float negative_slope,
                                     const int n) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp_offset = (blockIdx.x * WARPS_PER_BLOCK + wid) * WARP_SIZE;
    
    // Each thread processes 4 elements (float4)
    const unsigned int idx = warp_offset + lane;
    const unsigned int elements_per_thread = 4;
    const unsigned int total_elements = n / elements_per_thread;
    
    if (idx < total_elements) {
        // Use __ldg for read-only data
        float4 val = __ldg(&input[idx]);
        
        // Process each component
        val.x = val.x > 0.0f ? val.x : val.x * negative_slope;
        val.y = val.y > 0.0f ? val.y : val.y * negative_slope;
        val.z = val.z > 0.0f ? val.z : val.z * negative_slope;
        val.w = val.w > 0.0f ? val.w : val.w * negative_slope;
        
        // Store result
        output[idx] = val;
    }
}

__global__ void cleanup_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             const float negative_slope,
                             const int offset,
                             const int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        const float val = __ldg(&input[idx]);
        output[idx] = val > 0.0f ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int n = x.numel();
    const int vec_elements = n / 4;
    const int remainder = n % 4;
    
    // Process main data in chunks of float4
    if (vec_elements > 0) {
        const int total_warps = (vec_elements + WARP_SIZE - 1) / WARP_SIZE;
        const int blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        warp_leaky_relu_kernel<<<blocks, THREAD_COUNT>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            negative_slope,
            n
        );
    }
    
    // Handle remaining elements
    if (remainder > 0) {
        const int offset = vec_elements * 4;
        const int threads = 256;
        const int blocks = (remainder + threads - 1) / threads;
        
        cleanup_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            negative_slope,
            offset,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with warp optimizations");
}