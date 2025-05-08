#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float4 gelu_vec4(float4 x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* xi = ((float*)&x) + i;
        float* yi = ((float*)&result) + i;
        float x_cubed = *xi * *xi * *xi;
        float inner = sqrt_2_over_pi * (*xi + coeff * x_cubed);
        float tanh_val = tanhf(inner);
        *yi = 0.5f * *xi * (1.0f + tanh_val);
    }
    return result;
}

__global__ void gelu_kernel(const float* x, float* y, const int n) {
    extern __shared__ float4 shared_mem[];
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int vec_n = n / 4;
    
    // Process elements in chunks using shared memory
    for (int base = blockIdx.x * blockDim.x * 4; base < n; base += stride * 4) {
        int idx = base + tid * 4;
        if (idx < n) {
            // Load into shared memory
            shared_mem[tid] = idx + 4 <= n ? ((float4*)x)[idx/4] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();
        
        // Process data from shared memory
        if (idx < n) {
            float4 result = gelu_vec4(shared_mem[tid]);
            if (idx + 4 <= n) {
                ((float4*)y)[idx/4] = result;
            } else {
                // Handle edge case
                float* y_ptr = y + idx;
                float* result_ptr = (float*)&result;
                for (int i = 0; i < min(4, n - idx); i++) {
                    y_ptr[i] = result_ptr[i];
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 128; // Optimize for H100 occupancy
    const int max_blocks = 1024;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), max_blocks);
    
    // Allocate shared memory for float4 elements
    const int shared_mem_size = threads * sizeof(float4);
    
    gelu_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}