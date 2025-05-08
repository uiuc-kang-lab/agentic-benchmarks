#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float leaky_relu_op(float x, float negative_slope) {
    return x > 0.0f ? x : x * negative_slope;
}

__device__ __forceinline__ float4 process_vector4(float4 input, float negative_slope) {
    float4 result;
    result.x = leaky_relu_op(input.x, negative_slope);
    result.y = leaky_relu_op(input.y, negative_slope);
    result.z = leaky_relu_op(input.z, negative_slope);
    result.w = leaky_relu_op(input.w, negative_slope);
    return result;
}

__global__ void vectorized_leaky_relu_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    float negative_slope,
    int vector_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < vector_count) {
        const float4 in_vec = __ldg(&input[tid]);
        output[tid] = process_vector4(in_vec, negative_slope);
    }
}

__global__ void scalar_leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    int offset,
    int count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < count) {
        const int global_idx = offset + tid;
        output[global_idx] = leaky_relu_op(input[global_idx], negative_slope);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    const int total_elements = x.numel();
    const int vector_count = total_elements / 4;
    const int remainder = total_elements % 4;
    
    if (vector_count > 0) {
        const int threads = 512;
        const int blocks = (vector_count + threads - 1) / threads;
        
        vectorized_leaky_relu_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            negative_slope,
            vector_count
        );
    }
    
    if (remainder > 0) {
        const int threads = 512;
        const int blocks = (remainder + threads - 1) / threads;
        const int offset = vector_count * 4;
        
        scalar_leaky_relu_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            negative_slope,
            offset,
            remainder
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with modular device functions");
}