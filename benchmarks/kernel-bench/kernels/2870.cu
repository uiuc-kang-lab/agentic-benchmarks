#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ __forceinline__ T sigmoid_compute(T val) {
    T exp_val = expf(-val);
    return 1.0f / (1.0f + exp_val);
}

template<typename scalar_t>
__global__ void sigmoid_kernel_vectorized(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int64_t size) {
    constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid * vec_size; i < size; i += stride * vec_size) {
        float4 chunk;
        chunk.x = sigmoid_compute(static_cast<float>(input[i]));
        if (i + 1 < size) chunk.y = sigmoid_compute(static_cast<float>(input[i+1]));
        if (i + 2 < size) chunk.z = sigmoid_compute(static_cast<float>(input[i+2]));
        if (i + 3 < size) chunk.w = sigmoid_compute(static_cast<float>(input[i+3]));

        *reinterpret_cast<float4*>(&output[i]) = chunk;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        sigmoid_kernel_vectorized<scalar_t><<<blocks, threads>>>( 
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Sigmoid forward (CUDA)");
}