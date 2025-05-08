#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level vote primitives to minimize branch divergence
template <typename scalar_t>
__global__ void softplus_kernel_warp_vote(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Load the input value
    scalar_t x = input[idx];

    // Define full warp mask (assumes a warp size of 32)
    const unsigned int full_mask = 0xFFFFFFFF;
    
    // Use warp vote primitive to check if all threads in this warp satisfy the condition
    unsigned int high_mask = __ballot_sync(full_mask, x > static_cast<scalar_t>(20.0));
    unsigned int low_mask  = __ballot_sync(full_mask, x < static_cast<scalar_t>(-20.0));

    scalar_t res;
    // If all threads in the warp have x > 20.0, avoid computing exp overflow risk
    if (high_mask == full_mask) {
        res = x;
    } 
    // If all threads in the warp have x < -20.0, use exp(x) directly
    else if (low_mask == full_mask) {
        res = exp(x);
    } 
    else {
        // Fallback: compute each element's softplus with individual branch decisions
        if (x > static_cast<scalar_t>(20.0)) {
            res = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            res = exp(x);
        } else {
            res = log1p(exp(x));
        }
    }

    output[idx] = res;
}

// CUDA forward function wrapping the kernel launch
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_warp_vote<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
