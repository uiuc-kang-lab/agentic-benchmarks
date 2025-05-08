#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Use multiple threads per warp for better occupancy
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x; // Ensure coalesced memory access
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Grid-stride loop to handle multiple elements per thread
    while (idx < size) {
        // Softplus formula: f(x) = log(1 + exp(x))
        const scalar_t x = input[idx];
        if (x > 20.0) {
            output[idx] = x;
        } else if (x < -20.0) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
        idx += stride;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Use 512 threads per block for better occupancy on H100
    const int threads = 512;
    // Calculate optimal number of blocks based on SM count
    const int blocks = min(65535, (size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}