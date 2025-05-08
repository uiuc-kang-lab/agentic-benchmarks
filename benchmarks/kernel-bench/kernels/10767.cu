#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void subtract_shifted_cumsum(const scalar_t* cumsum, scalar_t* output, int64_t dim_size, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const int64_t slice = idx / dim_size;
    const int64_t pos = idx % dim_size;
    const scalar_t total = cumsum[slice * dim_size + dim_size - 1];
    
    // Use warp-shuffle to broadcast total within warp (32 threads share same slice)
    scalar_t warp_total = __shfl_sync(0xFFFFFFFF, total, (threadIdx.x % 32));
    const int64_t shifted_pos = (pos == 0) ? -1 : pos - 1;
    
    output[idx] = warp_total - (shifted_pos >= 0 ? cumsum[slice * dim_size + shifted_pos] : 0);
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    auto cumsum = x.cumsum(dim);
    auto output = torch::empty_like(x);
    
    const int64_t dim_size = x.size(dim);
    const int64_t num_elements = x.numel();
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        subtract_shifted_cumsum<scalar_t><<<blocks, threads>>>(
            cumsum.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            num_elements
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumsum with warp-broadcast total");
}