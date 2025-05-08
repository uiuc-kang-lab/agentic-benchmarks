#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ T device_relu(T val) {
    return val > 0 ? val : 0;
}

__device__ float4 relu_float4(const float4& vec) {
    return {
        device_relu(vec.x),
        device_relu(vec.y),
        device_relu(vec.z),
        device_relu(vec.w)
    };
}

template <typename scalar_t>
__global__ void relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t size) {
    
    constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(scalar_t);
    
    int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;
    int total_vectors = (size + VECTOR_SIZE - 1) / VECTOR_SIZE;
    int stride = blockDim.x * gridDim.x * VECTOR_SIZE;

    // Process complete vectors first
    while (global_idx < size - VECTOR_SIZE + 1) {
        float4 in_data = *reinterpret_cast<const float4*>(input + global_idx);
        *reinterpret_cast<float4*>(output + global_idx) = relu_float4(in_data);
        global_idx += stride;
    }

    // Handle remaining elements
    if (global_idx < size) {
        for(int i = 0; i < VECTOR_SIZE && (global_idx + i) < size; ++i) {
            output[global_idx + i] = device_relu(input[global_idx + i]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    int vectors_needed = (input.numel() + 3) / 4;
    const int blocks = (vectors_needed + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel", [&] {
        relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized ReLU with modular functions (CUDA)");
}