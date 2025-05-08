#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef struct __align__(16) {
    float x, y, z, w;
} float4;

__global__ void softsign_vec4_kernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    
    if (idx + 3 < num_elements) {
        float4 val;
        float4* x_vec = (float4*)&__ldg(x + idx);
        val = *x_vec;
        
        float4 result;
        result.x = val.x / (1.0f + fabsf(val.x));
        result.y = val.y / (1.0f + fabsf(val.y));
        result.z = val.z / (1.0f + fabsf(val.z));
        result.w = val.w / (1.0f + fabsf(val.w));
        
        ((float4*)(out + idx))[0] = result;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4; i++) {
            if (idx + i < num_elements) {
                float v = __ldg(x + idx + i);
                out[idx + i] = v / (1.0f + fabsf(v));
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    const int vec4_per_thread = 4;
    int threads = 512;  // Reduced threads since each handles 4 elements
    int blocks = (num_elements + (threads * vec4_per_thread) - 1) / (threads * vec4_per_thread);
    
    softsign_vec4_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                           out.data_ptr<float>(),
                                           num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with vectorized memory ops (CUDA)");
}