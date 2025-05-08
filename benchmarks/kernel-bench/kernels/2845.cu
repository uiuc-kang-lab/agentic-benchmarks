#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    // Grid-stride loop for better workload distribution
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) {
        
        // Use fast math intrinsics
        const float val = __fnegf(static_cast<float>(input[i]));
        const float result = 1.0f / (1.0f + __expf(val));
        output[i] = static_cast<scalar_t>(result);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Optimize block and grid dimensions
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = min((size + threads - 1) / threads, max_blocks);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Sigmoid forward (CUDA)");
}