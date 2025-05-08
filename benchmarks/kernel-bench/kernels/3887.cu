#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_unrolled(const float* __restrict__ x, 
                                       float* __restrict__ out, 
                                       const int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4
    int idx = tid * 4;
    const int step = stride * 4;
    
    // Main loop with manual unrolling for float4 processing
    while (idx + 3 < num_elements) {
        float4 in_val = *reinterpret_cast<const float4*>(x + idx);
        float4 out_val;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val;
            if (i == 0) val = in_val.x;
            else if (i == 1) val = in_val.y;
            else if (i == 2) val = in_val.z;
            else val = in_val.w;
            
            float result = val / (1.0f + fabsf(val));
            
            if (i == 0) out_val.x = result;
            else if (i == 1) out_val.y = result;
            else if (i == 2) out_val.z = result;
            else out_val.w = result;
        }
        
        *reinterpret_cast<float4*>(out + idx) = out_val;
        idx += step;
    }
    
    // Handle remaining elements
    #pragma unroll 4
    for (int i = idx; i < num_elements; i++) {
        if (i < num_elements) {
            float val = x[i];
            out[i] = val / (1.0f + fabsf(val));
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Optimize block and grid size for H100
    const int threads = 256;
    const int blocks = std::min(65535, (num_elements + (threads * 4) - 1) / (threads * 4));
    
    softsign_kernel_unrolled<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled Softsign activation (CUDA)");
}