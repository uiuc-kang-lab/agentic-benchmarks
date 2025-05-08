#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void subtract_shifted_cumsum_stride(const scalar_t* __restrict__ cumsum, scalar_t* __restrict__ output, int64_t dim_size, int64_t num_elements) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += grid_size) {
        const int64_t slice = i / dim_size;
        const int64_t pos = i % dim_size;
        const int64_t slice_start = slice * dim_size;
        const scalar_t total = cumsum[slice_start + dim_size - 1];

        // Directly use the loaded total; precomputed slice_start improves index arithmetic
        const int64_t shifted_pos = (pos == 0) ? -1 : pos - 1;
        const scalar_t prev = (shifted_pos >= 0 ? cumsum[slice_start + shifted_pos] : static_cast<scalar_t>(0));

        output[i] = total - prev;
    }
}

at::Tensor reverse_cumsum_optimized_stride(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    auto cumsum = x.cumsum(dim);
    auto output = torch::empty_like(x);
    
    const int64_t dim_size = x.size(dim);
    const int64_t num_elements = x.numel();
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        subtract_shifted_cumsum_stride<scalar_t><<<blocks, threads>>>(
            cumsum.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            num_elements
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized_stride, "Optimized reverse cumsum with stride loop");
}