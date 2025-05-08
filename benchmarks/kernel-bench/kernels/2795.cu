#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel minimizes warp divergence by precomputing the number of iterations each thread must perform
// in a grid-stride loop. With a branchless computation of the iteration count, the inner loop executes
// uniformly across all threads in a warp, reducing divergent control flow and improving performance.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Compute the number of iterations in a branchless manner.
    // If tid is out-of-bound, (tid < size) evaluates to 0, resulting in 0 iterations.
    int count = (tid < size) * (((size - 1 - tid) / stride) + 1);
    
    for (int i = 0; i < count; i++) {
        int idx = tid + i * stride;
        float x = static_cast<float>(input[idx]);
        float y = 1.0f / (1.0f + expf(-x));
        output[idx] = static_cast<scalar_t>(y);
    }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    constexpr int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) optimized for uniform control flow");
}
