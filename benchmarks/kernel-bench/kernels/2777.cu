#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel(const float4* __restrict__ input, 
                                float4* __restrict__ output,
                                float negative_slope,
                                int n) {
    __shared__ float4 shared_data[32];  // One float4 per thread in a warp
    
    const int tid = threadIdx.x;
    const int wid = tid / 32;  // Warp ID
    const int lane = tid % 32;  // Lane within warp
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread using grid stride loop
    for (int idx = gid; idx < n/4; idx += grid_stride) {
        // Load directly to registers
        float4 val = input[idx];
        
        // Process in registers
        float4 result;
        result.x = val.x > 0 ? val.x : val.x * negative_slope;
        result.y = val.y > 0 ? val.y : val.y * negative_slope;
        result.z = val.z > 0 ? val.z : val.z * negative_slope;
        result.w = val.w > 0 ? val.w : val.w * negative_slope;
        
        // Store directly from registers
        output[idx] = result;
    }
    
    // Handle remaining elements
    if (gid * 4 + 3 < n) {
        const float* in_float = reinterpret_cast<const float*>(input);
        float* out_float = reinterpret_cast<float*>(output);
        
        int base = (n/4)*4;
        int idx = base + gid;
        
        if (idx < n) {
            float val = in_float[idx];
            out_float[idx] = val > 0 ? val : val * negative_slope;
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 128;  // Multiple of warp size
    const int blocks = std::min(65535, (n + threads*4 - 1) / (threads*4));
    
    leaky_relu_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        negative_slope,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}