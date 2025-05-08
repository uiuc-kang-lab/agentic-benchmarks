#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel with manually unrolled grid-stride loop
__global__ void softsign_kernel_unrolled(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process 4 elements per iteration using manual unrolling
    for (int i = idx; i < num_elements; i += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int index = i + j * stride;
            if (index < num_elements) {
                float val = x[index];
                out[index] = val / (1.0f + fabsf(val));
            }
        }
    }
}

// Host function
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_unrolled<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with unrolled loops (CUDA)");
}
