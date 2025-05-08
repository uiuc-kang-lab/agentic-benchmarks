#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

template <typename scalar_t>
__global__ void adaptive_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size,
    const bool use_vectorized) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (use_vectorized) {
        const int vec4_size = size / 4;
        const float4* input4 = reinterpret_cast<const float4*>(input);
        float4* output4 = reinterpret_cast<float4*>(output);
        
        // Vectorized processing
        for (int i = tid; i < vec4_size; i += stride) {
            float4 in4 = input4[i];
            output4[i] = tanh_vec4<scalar_t>(in4);
        }
        
        // Handle remaining elements
        const int remaining_start = vec4_size * 4;
        for (int i = remaining_start + tid; i < size; i += stride) {
            output[i] = tanhf(input[i]);
        }
    } else {
        // Non-vectorized processing for small tensors
        for (int i = tid; i < size; i += stride) {
            output[i] = tanhf(input[i]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Choose processing mode based on tensor size
    const bool use_vectorized = (size >= 1024); // Threshold for vectorization
    const int threads = 256;
    const int blocks = use_vectorized ? 
        (size / 4 + threads - 1) / threads :
        (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_tanh_kernel", ([&] {
        adaptive_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size,
            use_vectorized
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Tanh forward (CUDA)");
}