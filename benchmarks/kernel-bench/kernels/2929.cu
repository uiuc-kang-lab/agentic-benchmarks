#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reinterpret_cast<float*>(&result)[i] = tanhf(reinterpret_cast<const float*>(&val)[i]);
    }
    return result;
}

template <typename scalar_t>
__global__ void tanh_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    __shared__ float4 shared_data[32];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * (blockDim.x * 4) + tid;
    const int stride = blockDim.x * gridDim.x * 4;
    const int vec4_size = size / 4;
    
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    #pragma unroll 4
    for (int i = idx; i < vec4_size; i += stride) {
        if (i < vec4_size) {
            shared_data[tid] = input4[i];
        }
        __syncthreads();
        
        if (i < vec4_size) {
            output4[i] = tanh_vec4<scalar_t>(shared_data[tid]);
        }
        __syncthreads();
    }
    
    const int remaining_start = vec4_size * 4;
    if (remaining_start < size) {
        for (int i = remaining_start + tid; i < size; i += blockDim.x) {
            output[i] = tanhf(input[i]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int elements_per_thread = 4;
    const int blocks = min(
        256,
        (input.numel() + threads * elements_per_thread - 1) / (threads * elements_per_thread)
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_optimized", ([&] {
        tanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}