#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for sigmoid computation
__device__ __forceinline__ float sigmoid_compute(float val) {
    float exp_val = expf(-val);
    return 1.0f / (1.0f + exp_val);
}

// Kernel using modular device function for sigmoid
template<typename scalar_t>
__global__ void sigmoid_kernel_modular(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int64_t size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        output[i] = static_cast<scalar_t>(sigmoid_compute(static_cast<float>(input[i])));
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        sigmoid_kernel_modular<scalar_t><<<blocks, threads>>>( 
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Sigmoid forward (CUDA)");
}