#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t size, int64_t /*dim_stride*/) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Compute the reversed index: we want out[r_idx] = sum(x[r_idx:])
        int r_idx = size - 1 - idx;
        float sum = 0.0f;
        // Perform the cumulative sum from r_idx to the end
        for (int j = r_idx; j < size; ++j) {
            sum += x[j];
        }
        out[r_idx] = sum;
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    at::Tensor out = at::zeros_like(x);
    int64_t size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Calculate strides for dimension
    int64_t dim_stride = 1;
    for (int i = 0; i < dim; ++i) {
        dim_stride *= x.size(i);
    }

    // Launch kernel
    reverse_cumsum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        dim_stride
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Reverse cumulative sum with memory coalescing (CUDA)");
}