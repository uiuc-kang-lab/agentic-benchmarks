#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized sigmoid kernel combining multi-element processing and memory coalescing
// Processes multiple elements per thread with striding to ensure high occupancy.

template <typename scalar_t>
__global__ void optimized_sigmoid_kernel(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           const int64_t size) {
    // Calculate the unique thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements by striding through the input array
    for (int i = tid; i < size; i += stride) {
        // Load element and compute sigmoid (1 / (1 + exp(-x)))
        float val = static_cast<float>(input[i]);
        float exp_val = expf(-val);
        float sigmoid = 1.0f / (1.0f + exp_val);
        output[i] = static_cast<scalar_t>(sigmoid);
    }
}

// Forward function to launch the kernel
torch::Tensor forward(torch::Tensor input) {
    // Create an output tensor of the same size
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Kernel launch configuration
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Dispatch the kernel based on the input data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        optimized_sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    });

    return output;
}

// Bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Sigmoid forward (CUDA) with multi-element processing for improved memory coalescing");
}
