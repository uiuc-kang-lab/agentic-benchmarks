#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float compute_softsign(float x) {
    return x / (1.0f + fabsf(x));
}

__device__ __forceinline__ float4 process_vector4(float4 input) {
    float4 output;
    output.x = compute_softsign(input.x);
    output.y = compute_softsign(input.y);
    output.z = compute_softsign(input.z);
    output.w = compute_softsign(input.w);
    return output;
}

__device__ __forceinline__ void process_remainder(const float* __restrict__ x, 
                                                float* __restrict__ out,
                                                int idx, 
                                                int num_elements) {
    if (idx < num_elements) {
        out[idx] = compute_softsign(x[idx]);
    }
}

__global__ void softsign_kernel_modular(const float* __restrict__ x, 
                                      float* __restrict__ out, 
                                      int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_elements = num_elements / 4 * 4;
    
    // Process blocks of 4 elements
    for (int idx = tid * 4; idx < vector_elements; idx += blockDim.x * gridDim.x * 4) {
        float4 input = *reinterpret_cast<const float4*>(&x[idx]);
        float4 output = process_vector4(input);
        *reinterpret_cast<float4*>(&out[idx]) = output;
    }
    
    // Handle remaining elements
    for (int idx = vector_elements + tid; idx < num_elements; idx += blockDim.x * gridDim.x) {
        process_remainder(x, out, idx, num_elements);
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    int threads = 256;
    int blocks = std::min(65535, (num_elements + threads - 1) / threads);
    
    softsign_kernel_modular<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with modular device functions (CUDA)");
}