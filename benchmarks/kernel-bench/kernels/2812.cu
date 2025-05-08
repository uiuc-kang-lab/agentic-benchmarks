#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;

// Modular device function: type-specific sigmoid implementation
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_val(scalar_t x);

template <>
__device__ __forceinline__ float sigmoid_val<float>(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <>
__device__ __forceinline__ double sigmoid_val<double>(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Modular device function to process a fixed number of elements per thread
template <typename scalar_t>
__device__ __forceinline__ void process_elements(const scalar_t* __restrict__ input,
                                                    scalar_t* __restrict__ output,
                                                    int64_t start_idx,
                                                    int elements,
                                                    int64_t size) {
    #pragma unroll
    for (int i = 0; i < elements; i++) {
        int64_t idx = start_idx + i;
        if (idx < size) {
            scalar_t val = input[idx];
            output[idx] = sigmoid_val<scalar_t>(val);
        }
    }
}

// Main kernel using grid-stride loop, processing ELEMENTS_PER_THREAD per thread
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  int64_t size) {
    // Each block processes a chunk of (THREADS * ELEMENTS_PER_THREAD) elements
    const int64_t chunk = THREADS * ELEMENTS_PER_THREAD;
    
    // Calculate the starting index for each thread's chunk
    for (int64_t base = blockIdx.x * chunk + threadIdx.x * ELEMENTS_PER_THREAD;
         base < size;
         base += gridDim.x * chunk) {
        process_elements<scalar_t>(input, output, base, ELEMENTS_PER_THREAD, size);
    }
}

// Forward function to launch the kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Calculate the number of blocks needed
    const int blocks = (size + THREADS * ELEMENTS_PER_THREAD - 1) / (THREADS * ELEMENTS_PER_THREAD);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();

        sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular and optimized Sigmoid forward (CUDA)");
}
