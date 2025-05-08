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
__global__ void tanh_kernel_vectorized_unrolled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int vec4_elements) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process input using float4 with manual unrolling
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Manual 4x unroll of the main loop
    #pragma unroll 4
    for (int i = tid; i < vec4_elements - 3; i += stride * 4) {
        float4 in4_0 = input4[i];
        float4 in4_1 = input4[i + stride];
        float4 in4_2 = input4[i + stride * 2];
        float4 in4_3 = input4[i + stride * 3];
        
        // Process in registers
        float4 out4_0 = tanh_vec4<scalar_t>(in4_0);
        float4 out4_1 = tanh_vec4<scalar_t>(in4_1);
        float4 out4_2 = tanh_vec4<scalar_t>(in4_2);
        float4 out4_3 = tanh_vec4<scalar_t>(in4_3);
        
        // Write back results
        output4[i] = out4_0;
        output4[i + stride] = out4_1;
        output4[i + stride * 2] = out4_2;
        output4[i + stride * 3] = out4_3;
    }
    
    // Handle remaining elements
    for (int i = tid + (vec4_elements/4)*4*stride; i < vec4_elements; i += stride) {
        float4 in4 = input4[i];
        output4[i] = tanh_vec4<scalar_t>(in4);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int vec4_size = input.numel() / 4;
    const int threads = 256;
    const int blocks = std::min(65535, (vec4_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_vectorized_unrolled", ([&] {
        tanh_kernel_vectorized_unrolled<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            vec4_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward vectorized unrolled (CUDA)");
}