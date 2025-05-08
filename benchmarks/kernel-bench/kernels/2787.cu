#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes the Sigmoid activation, using a tuned block size based on performance experiments.
// We use a grid-stride loop to cover all elements in the tensor. The block size is set to 512.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        // Compute sigmoid using full precision float computations
        float val = static_cast<float>(input[i]);
        float exp_val = expf(-val);
        float res = 1.0f / (1.0f + exp_val);
        output[i] = static_cast<scalar_t>(res);
    }
}

// Forward function to be called from Python
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Experimentally tuned block size based on NVIDIA H100 characteristics
    constexpr int THREADS = 512; 
    const int blocks = (size + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();

        sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) tuned block size");
}
