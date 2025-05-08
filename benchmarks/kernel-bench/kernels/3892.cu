#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float softsign_compute(float x) {
    return x / (1.0f + fabsf(x));
}

__global__ void softsign_kernel_unrolled(const float* __restrict__ x, 
                                       float* __restrict__ out, 
                                       const int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 8;
    
    // Process 8 elements per thread in the main loop
    #pragma unroll
    for (int i = tid; i < num_elements - (unroll_factor - 1); i += stride * unroll_factor) {
        float vals[unroll_factor];
        float results[unroll_factor];
        
        // Load 8 elements
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            if (i + j * stride < num_elements) {
                vals[j] = x[i + j * stride];
            }
        }
        
        // Process 8 elements
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            if (i + j * stride < num_elements) {
                results[j] = softsign_compute(vals[j]);
            }
        }
        
        // Store 8 elements
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            if (i + j * stride < num_elements) {
                out[i + j * stride] = results[j];
            }
        }
    }
    
    // Handle remaining elements
    const int remaining_start = num_elements - (num_elements % (stride * unroll_factor));
    for (int i = remaining_start + tid; i < num_elements; i += stride) {
        out[i] = softsign_compute(x[i]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Optimize thread block size for H100
    const int threads = 256;
    const int blocks = std::min(65535, (num_elements + threads - 1) / threads);
    
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