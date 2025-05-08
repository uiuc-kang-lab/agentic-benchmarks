#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int STRIDE>
__global__ void leaky_relu_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    float alpha,
    int num_elements) {
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < num_elements / 4; 
        i += blockDim.x * gridDim.x * STRIDE) {
        
        float4 vec = ((const float4*)__builtin_assume_aligned(input_vec, 16))[i];
        float4 result;
        
        result.x = __fmul_rn(vec.x * (vec.x > 0.0f), 1.0f - alpha) + vec.x * alpha;
        result.y = __fmul_rn(vec.y * (vec.y > 0.0f), 1.0f - alpha) + vec.y * alpha;
        result.z = __fmul_rn(vec.z * (vec.z > 0.0f), 1.0f - alpha) + vec.z * alpha;
        result.w = __fmul_rn(vec.w * (vec.w > 0.0f), 1.0f - alpha) + vec.w * alpha;
        
        ((float4*)__builtin_assume_aligned(output_vec, 16))[i] = result;
    }

    // Handle remaining elements (less than 4)
    int remainder_start = (num_elements / 4) * 4;
    int idx = remainder_start + blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_elements) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : val * alpha;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    
    const int threads = 256;
    const int vector_stride = 4; // Process 4 elements per vector
    const int blocks = (n + threads * vector_stride - 1) / (threads * vector_stride);
    
    leaky_relu_vectorized<4><<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), 
        negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with coalesced vector accesses");
}