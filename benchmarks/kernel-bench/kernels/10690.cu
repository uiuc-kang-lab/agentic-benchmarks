#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum along a given dimension.
// This version aims to distribute workloads evenly across threads and blocks.

template <typename scalar_t>
__global__ void reverse_cumsum_balanced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t n,        // size of the cumsum dimension
    int64_t outer) {  // number of slices (all dimensions except the one being cumsum-ed)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Each thread processes multiple elements using a grid-stride loop
    for (int64_t idx = tid; idx < outer * n; idx += total_threads) {
        int64_t row = idx / n;
        int64_t col = idx % n;
        int64_t offset = row * n;

        // Compute reverse cumulative sum for this element
        scalar_t cum = 0;
        for (int64_t j = n - 1; j >= col; j--) {
            cum += input[offset + j];
        }
        output[offset + col] = cum;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    const int threads = 256;
    const int blocks = (outer * n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_balanced", ([&] {
        reverse_cumsum_balanced_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n, outer);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Balanced reverse cumulative sum (CUDA)");
}