#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel implements GELU activation using a grid-stride loop that is manually unrolled
// to reduce loop overhead. The inner loop is unrolled with an unroll factor of 8 using #pragma unroll.

__global__ void gelu_kernel_unrolled(const float* __restrict__ x, float* __restrict__ y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int unroll_factor = 8;

    // Compute limit such that i + (unroll_factor - 1) * stride < n
    int limit = n - (unroll_factor - 1) * stride;

    // Manually unrolled loop to process unroll_factor elements per iteration
    for (int i = tid; i < limit; i += stride * unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            int index = i + j * stride;
            float val = x[index];
            float x_cubed = val * val * val;
            float inner = (val + 0.044715f * x_cubed) * 0.7978845608f;  // sqrt(2/pi) constant
            y[index] = 0.5f * val * (1.0f + tanhf(inner));
        }
    }

    // Process any remaining elements
    for (int i = limit; i < n; i += stride) {
        float val = x[i];
        float x_cubed = val * val * val;
        float inner = (val + 0.044715f * x_cubed) * 0.7978845608f;
        y[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    gelu_kernel_unrolled<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with manual loop unrolling");
}
