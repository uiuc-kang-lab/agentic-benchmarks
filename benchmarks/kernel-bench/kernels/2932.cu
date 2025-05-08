#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory to optimize memory access

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
__global__ void tanh_kernel_shared_memory(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    extern __shared__ float4 shared_input[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;
    
    // Load input into shared memory
    for (int i = idx; i < vec4_size; i += stride) {
        shared_input[threadIdx.x] = reinterpret_cast<const float4*>(input)[i];
        __syncthreads();

        // Perform computation using shared memory
        float4 in4 = shared_input[threadIdx.x];
        float4 out4 = tanh_vec4<scalar_t>(in4);
        reinterpret_cast<float4*>(output)[i] = out4;
        __syncthreads();
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    for (int i = remaining_start + idx; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_shared_memory", ([&] {
        tanh_kernel_shared_memory<scalar_t><<<blocks, threads, threads * sizeof(float4)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with shared memory optimization (CUDA)");
}