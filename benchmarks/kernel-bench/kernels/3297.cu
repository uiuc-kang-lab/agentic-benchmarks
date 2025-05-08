#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to process main bulk using 128-bit aligned vectorized loads with __ldg()
__global__ void swish_ldg_vectorized_kernel(const float4* __restrict__ x, float4* __restrict__ y, int64_t n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n4) {
        // Use __ldg() for read-only load of 128-bit data
        float4 input_val = __ldg(&x[idx]);
        float4 output_val;
        float* in_ptr = reinterpret_cast<float*>(&input_val);
        float* out_ptr = reinterpret_cast<float*>(&output_val);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v = in_ptr[i];
            float sigmoid = 1.0f / (1.0f + expf(-v));
            out_ptr[i] = v * sigmoid;
        }
        y[idx] = output_val;
        idx += stride;
    }
}

// Kernel to handle remainder elements when total elements is not a multiple of 4
__global__ void swish_ldg_remainder_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t start, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = start + idx; i < n; i += stride) {
        float v = __ldg(&x[i]);
        float sigmoid = 1.0f / (1.0f + expf(-v));
        y[i] = v * sigmoid;
    }
}

// CUDA forward function using __ldg() for optimized global memory access and 128-bit alignment
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    int64_t n4 = n / 4;  // process in chunks of 4 floats (128 bits)

    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;

    swish_ldg_vectorized_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4
    );

    int64_t remainder = n % 4;
    if (remainder > 0) {
        int64_t start = n4 * 4;
        const int blocks_remainder = 1;  // small remainder can be processed in one block
        swish_ldg_remainder_kernel<<<blocks_remainder, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            start,
            n
        );
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation using __ldg() and 128-bit aligned memory accesses (CUDA)");
}
