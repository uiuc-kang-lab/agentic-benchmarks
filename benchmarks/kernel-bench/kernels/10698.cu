#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined Kernel to compute and optimize reverse cumulative sum along a given dimension.
// Uses efficient memory access and thread allocation strategies.

template <typename scalar_t>
__global__ void optimized_reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer,    // number of slices (all dimensions except the one being cumsum-ed)
    int64_t n,        // size of the cumsum dimension
    int ndim,         // total number of dimensions of the tensor
    int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer) return;

    // Each thread computes the reverse cumulative sum for one slice, optimizing memory access
    int64_t offset = idx * n;

    scalar_t cum = scalar_t(0);
    for (int64_t i = n - 1; i >= 0; --i) {
        cum += input[offset + i];
        output[offset + i] = cum;
    }
}

at::Tensor optimized_reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Prepare output tensor
    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Set kernel launch parameters
    const int threads = 256;
    const int blocks = (outer + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_reverse_cumsum_cuda", ([&] {
        optimized_reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            n,
            ndim,
            dim);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_reverse_cumsum, "Optimized Reverse Cumulative Sum (CUDA)");
}
