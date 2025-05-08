#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a grid-stride loop, which maps threads efficiently to the problem domain.
// Each thread computes multiple elements separated by the full grid stride, ensuring that all elements are covered,
// even when the number of elements is much larger than the number of threads. This approach leverages
// correct use of threadIdx, blockIdx, blockDim, and gridDim for optimal occupancy and reduced kernel launch overhead.
__global__ void elu_kernel_grid_stride(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < n; idx += stride) {
        float val = __ldg(&x[idx]);
        out[idx] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
    }
}

// The host function sets up the kernel launch using a 1D grid-stride loop kernel.
// This ensures efficient mapping of GPU threads across the entire data set irrespective of tensor size.
// The kernel returns the correct ELU activation by processing every element in the input tensor.

torch::Tensor elu_cuda_grid_stride(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_grid_stride<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_grid_stride, "ELU activation with grid-stride loop (CUDA)");
}
