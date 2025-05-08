#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized and optimized kernel with grid-stride loops for large workloads

template <typename scalar_t>
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

__global__ void tanh_kernel_stride_loop(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;

    // Use grid-stride loops to handle larger workloads
    for (int i = idx; i < vec4_size; i += stride) {
        // Process 4 elements at a time using float4
        float4 in4 = reinterpret_cast<const float4*>(input)[i];
        reinterpret_cast<float4*>(output)[i] = tanh_vec4(in4);
    }
    
    // Handle remaining elements
    for (int i = 4 * vec4_size + idx; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}


torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads * 4 - 1) / (threads * 4);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_stride_loop", ([&] {
        tanh_kernel_stride_loop<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with stride loop optimization (CUDA)");
}
