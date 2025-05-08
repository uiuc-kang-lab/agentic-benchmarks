#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const scalar_t x = input[idx];
        
        // Get warp information
        const unsigned int mask = __activemask();
        const int lane = threadIdx.x & (warpSize - 1);
        
        // Fast path check for entire warp
        const bool is_high = x > 20.0f;
        const bool is_low = x < -20.0f;
        const unsigned int high_mask = __ballot_sync(mask, is_high);
        const unsigned int low_mask = __ballot_sync(mask, is_low);
        
        // If all threads in warp are in same range, use fast path
        if (high_mask == mask) {
            output[idx] = x;  // All high values
        } else if (low_mask == mask) {
            output[idx] = exp(x);  // All low values
        } else {
            // Mixed case - use optimized branch
            if (x > 0) {
                output[idx] = x + log1p(exp(-x));  // More numerically stable for x > 0
            } else {
                output[idx] = log1p(exp(x));
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
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