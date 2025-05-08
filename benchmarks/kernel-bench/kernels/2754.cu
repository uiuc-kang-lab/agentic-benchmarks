#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses warp-level primitives (__ballot_sync and __all_sync) to check if all active threads in a warp
// have values that are uniformly positive or uniformly non-positive. In such cases, the branch can be resolved
// without divergence. A grid-stride loop ensures all elements are processed without using shared memory.
__global__ void warp_ballot_leaky_relu_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Full warp mask for 32 threads
    const unsigned int fullMask = 0xffffffff;

    for (; idx < n; idx += stride) {
        float val = x[idx];

        // Each thread in the warp evaluates if its index is valid.
        unsigned int active_mask = __ballot_sync(fullMask, idx < n);

        // Check if all active lanes in the warp are positive or non-positive
        bool all_positive = __all_sync(active_mask, val > 0.0f);
        bool all_negative = __all_sync(active_mask, val <= 0.0f);

        if (all_positive) {
            // All active lanes are positive, avoid branch divergence
            out[idx] = val;
        } else if (all_negative) {
            // All active lanes are non-positive, process uniformly
            out[idx] = val * negative_slope;
        } else {
            // Mixed values within the warp, evaluate per element
            out[idx] = (val > 0.0f) ? val : (val * negative_slope);
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    warp_ballot_leaky_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with warp-level primitives");
}
