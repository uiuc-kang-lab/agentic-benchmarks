#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__forceinline__ __device__ float sigmoid_compute(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename scalar_t>
__forceinline__ __device__ void process_element(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              const int idx) {
    float val = static_cast<float>(input[idx]);
    output[idx] = static_cast<scalar_t>(sigmoid_compute<scalar_t>(val));
}

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        process_element<scalar_t>(input, output, idx);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}