#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Store the frequently used constant 1.0f in constant memory
__constant__ float c_one = 1.0f;

__global__ void softsign_kernel_const(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to process all elements
    for (; idx < num_elements; idx += stride) {
        float val = x[idx];
        out[idx] = val / (c_one + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    const int threads = 256;  // Reduced thread count for better occupancy
    const int max_blocks = 32768;  // Limit maximum blocks to avoid oversubscription
    const int blocks = min((num_elements + threads - 1) / threads, max_blocks);
    
    softsign_kernel_const<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with constant memory optimization (CUDA)");
}
