#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return 1.0f / (1.0f + __expf(-x));
}

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int64_t size) {
    constexpr int vec_size = sizeof(float4)/sizeof(scalar_t);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vectorized processing for aligned 128-bit accesses
    if constexpr (std::is_same_v<scalar_t, float>) {
        float4* out_vec = reinterpret_cast<float4*>(output);
        int vectorizable_size = (size / vec_size) * vec_size;
        
        #pragma unroll
        for(int i = tid * vec_size; i < vectorizable_size; i += blockDim.x * gridDim.x * vec_size) {
            const float4 data = __ldg(reinterpret_cast<const float4*>(input + i));
            float4 result;
            result.x = sigmoid(data.x);
            result.y = sigmoid(data.y);
            result.z = sigmoid(data.z);
            result.w = sigmoid(data.w);
            out_vec[i/vec_size] = result;
        }
    }
    
    // Scalar tail handling using read-only cache
    int vectorizable_size = (size / vec_size) * vec_size;
    for(int i = tid + vectorizable_size; i < size; i += blockDim.x * gridDim.x) {
        output[i] = sigmoid(__ldg(input + i));
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    int blocks = (size + threads * 4) / (threads * 4);
    blocks = std::min(blocks, 108);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                     output.data_ptr<scalar_t>(),
                                                     size);
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Sigmoid with LDG & aligned 128b");
}