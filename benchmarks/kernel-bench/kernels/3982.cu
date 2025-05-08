#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory for frequently used values
__constant__ float c_one = 1.0f;
__constant__ float4 c_ones = {1.0f, 1.0f, 1.0f, 1.0f};

__global__ void softsign_constant_kernel(const float4* __restrict__ input,
                                       float4* __restrict__ output,
                                       const int n_vectors) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < n_vectors; idx += stride) {
        float4 in = input[idx];
        float4 result;
        
        // Use constant memory for the denominator addition
        result.x = in.x / (c_one + fabsf(in.x));
        result.y = in.y / (c_one + fabsf(in.y));
        result.z = in.z / (c_one + fabsf(in.z));
        result.w = in.w / (c_one + fabsf(in.w));
        
        output[idx] = result;
    }
}

__global__ void softsign_remainder_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int offset,
                                        const int total_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < total_elements) {
        // Use constant memory for single float operations
        output[idx] = input[idx] / (c_one + fabsf(input[idx]));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    const int num_elements = x.numel();
    const int num_vectors = num_elements / 4;
    const int remainder = num_elements % 4;
    
    // Optimize thread/block count for H100
    constexpr int threads = 256;
    constexpr int max_blocks = 512;
    const int blocks = min(max_blocks, (num_vectors + threads - 1) / threads);
    
    if (num_vectors > 0) {
        softsign_constant_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            num_vectors
        );
    }
    
    if (remainder > 0) {
        const int remainder_offset = num_vectors * 4;
        const int remainder_blocks = (remainder + threads - 1) / threads;
        
        softsign_remainder_kernel<<<remainder_blocks, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            remainder_offset,
            num_elements
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign using constant memory (CUDA)");
}