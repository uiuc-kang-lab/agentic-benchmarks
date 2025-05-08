#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int vec_size = 4;
    
    // Process 4 elements per thread
    for (int i = tid * vec_size; i < size; i += stride * vec_size) {
        float4* out_vec = (float4*)&output[i];
        const float4* in_vec = (const float4*)&input[i];
        
        if (i + vec_size <= size) {
            float4 val = *in_vec;
            val.x = val.x > 0 ? val.x : 0;
            val.y = val.y > 0 ? val.y : 0;
            val.z = val.z > 0 ? val.z : 0;
            val.w = val.w > 0 ? val.w : 0;
            *out_vec = val;
        } else {
            // Handle boundary case
            for (int j = 0; j < min(vec_size, size - i); j++) {
                output[i + j] = input[i + j] > 0 ? input[i + j] : 0;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = min(65535, (input.numel() + (threads * 4) - 1) / (threads * 4));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_vectorized", ([&] {
        relu_kernel_vectorized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}