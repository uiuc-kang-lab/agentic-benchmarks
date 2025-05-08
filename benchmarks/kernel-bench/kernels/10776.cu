#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void subtract_shifted_cumsum(const scalar_t* cumsum, scalar_t* output, int64_t dim_size, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const int64_t slice = idx / dim_size;
    const int64_t pos = idx % dim_size;
    const scalar_t total = cumsum[slice * dim_size + dim_size - 1];
    
    // Warp stride access with coalesced memory pattern
    scalar_t warp_total = __shfl_sync(0xFFFFFFFF, total, (threadIdx.x % 32));
    const int64_t shifted_pos = (pos == 0) ? -1 : pos - 1;
    
    output[idx] = warp_total - (shifted_pos >= 0 ? cumsum[slice * dim_size + shifted_pos] : 0);
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    // Check contiguity and transpose if needed
    auto sizes = x.sizes().vec();
    auto strides = x.strides().vec();
    auto min_stride = *std::min_element(strides.begin(), strides.end());
    
    bool permuted = false;
    std::vector<int64_t> permute_dims(x.dim());
    std::iota(permute_dims.begin(), permute_dims.end(), 0);
    
    if (strides[dim] != min_stride) {
        std::swap(permute_dims[dim], permute_dims.back());
        x = x.permute(permute_dims).contiguous();
        permuted = true;
        dim = x.dim() - 1;
    } else {
        x = x.contiguous();
    }

    auto cumsum = x.cumsum(dim);
    auto output = torch::empty_like(cumsum);
    
    const int64_t dim_size = cumsum.size(dim);
    const int64_t num_elements = cumsum.numel();
    
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
    
    if (permuted) {
        output = output.permute(permute_dims);
    }
    
    return output.contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumsum with memory coalescing transpose");
}