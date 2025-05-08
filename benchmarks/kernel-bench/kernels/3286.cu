#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel applying Swish activation with manual loop unrolling
__global__ void swish_unrolled_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Unroll the loop by a factor of 4 to reduce loop overhead
    for (int i = idx; i < n; i += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int pos = i + j * stride;
            if (pos < n) {
                float val = x[pos];
                float sigmoid = 1.0f / (1.0f + expf(-val));
                y[pos] = val * sigmoid;
            }
        }
    }
}

// C++ interface to launch the CUDA kernel
torch::Tensor swish_unrolled_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_unrolled_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_unrolled_forward, "Swish activation forward pass with loop unrolling (CUDA)");
}
