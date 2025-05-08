#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Softsign activation: out = x / (1 + |x|)
// This kernel uses a grid-stride loop that has been unrolled by a factor of 4
// to reduce loop overhead and improve instruction-level parallelism.

__global__ void softsign_kernel_unroll(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Unroll by a factor of 4
    #pragma unroll
    for (int i = idx; i < n; i += stride * 4) {
        if (i < n) {
            float val = in[i];
            out[i] = val / (1.0f + fabsf(val));
        }
        if (i + stride < n) {
            float val = in[i + stride];
            out[i + stride] = val / (1.0f + fabsf(val));
        }
        if (i + 2 * stride < n) {
            float val = in[i + 2 * stride];
            out[i + 2 * stride] = val / (1.0f + fabsf(val));
        }
        if (i + 3 * stride < n) {
            float val = in[i + 3 * stride];
            out[i + 3 * stride] = val / (1.0f + fabsf(val));
        }
    }
}

// Host function wrapping the kernel
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_unroll<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with unrolled grid-stride loop (CUDA)");
}
