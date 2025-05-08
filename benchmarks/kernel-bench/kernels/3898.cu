#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hybrid kernel: uses vectorized float4 operations for the bulk of data and
// scalar processing for leftover elements. This removes unnecessary shared memory
// overhead while still supporting any tensor size.
__global__ void softsign_kernel_hybrid(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int num_elements) {
    // Calculate how many complete float4 groups we have
    int vector_elems = num_elements / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process main chunk with vectorized loads/stores
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);
    for (int i = tid; i < vector_elems; i += stride) {
        float4 in_val = x_vec[i];
        float4 result;
        result.x = in_val.x / (1.0f + fabsf(in_val.x));
        result.y = in_val.y / (1.0f + fabsf(in_val.y));
        result.z = in_val.z / (1.0f + fabsf(in_val.z));
        result.w = in_val.w / (1.0f + fabsf(in_val.w));
        y_vec[i] = result;
    }

    // Process remaining elements if num_elements is not a multiple of 4
    int remainder_start = vector_elems * 4;
    for (int i = tid; i < (num_elements - remainder_start); i += stride) {
        int idx = remainder_start + i;
        float val = x[idx];
        y[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    // Choose kernel configuration based on total number of elements
    const int threads = 256;
    const int blocks = std::min(65535, (num_elements + threads - 1) / threads);

    softsign_kernel_hybrid<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Softsign activation (CUDA)");
}
