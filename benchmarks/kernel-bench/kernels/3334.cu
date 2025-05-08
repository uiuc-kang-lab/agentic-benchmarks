#include <torch/extension.h>

// Optimized Swish activation kernel with manual loop unrolling and __restrict__ qualifiers
__global__ void swish_opt_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Compute global thread index and stride using grid-stride loop pattern
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Loop variable
    int64_t i = idx;

    // Unroll factor of 4 for higher throughput
    #pragma unroll 4
    for (; i + 3 * stride < n; i += 4 * stride) {
        // Load four elements from global memory
        float a = x[i];
        float b = x[i + stride];
        float c = x[i + 2 * stride];
        float d = x[i + 3 * stride];

        // Compute swish activation using fast math __expf
        float s_a = 1.0f / (1.0f + __expf(-a));
        float s_b = 1.0f / (1.0f + __expf(-b));
        float s_c = 1.0f / (1.0f + __expf(-c));
        float s_d = 1.0f / (1.0f + __expf(-d));

        // Write results back to global memory
        y[i] = a * s_a;
        y[i + stride] = b * s_b;
        y[i + 2 * stride] = c * s_c;
        y[i + 3 * stride] = d * s_d;
    }

    // Handle remaining elements with a tail loop
    for (; i < n; i += stride) {
        float a = x[i];
        float s_a = 1.0f / (1.0f + __expf(-a));
        y[i] = a * s_a;
    }
}

// Forward function that launches the optimized kernel
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Use 256 threads per block and compute the number of blocks
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_opt_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation forward pass (CUDA)");
}
