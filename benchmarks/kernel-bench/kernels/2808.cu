#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused Sigmoid Kernel: Combines grid-stride loop for memory coalescing with inline sigmoid computation
template <typename scalar_t>
__forceinline__ __device__ scalar_t sigmoid_val(scalar_t x) {
    // Convert to float for computation and then cast back
    float fx = static_cast<float>(x);
    return static_cast<scalar_t>(1.0f / (1.0f + expf(-fx)));
}

// Kernel that processes multiple elements per thread
// for improved coalescing and performance
template <typename scalar_t>
__global__ void fused_sigmoid_kernel(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       const int64_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < size; idx += stride) {
        // Inline arithmetic avoids function call overhead
        float val = static_cast<float>(input[idx]);
        // Compute sigmoid
        float exp_val = expf(-val);
        output[idx] = static_cast<scalar_t>(1.0f / (1.0f + exp_val));
    }
}

// Host function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Configure launch parameters
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        fused_sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Sigmoid forward (CUDA) optimized with grid-stride loops and inline computation");
}
