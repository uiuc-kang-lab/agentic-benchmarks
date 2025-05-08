#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int64_t dim_size, int64_t stride_dim, int64_t num_slices) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t base = idx * dim_size;
    
    while (idx < num_slices) {
        scalar_t sum = 0;
        // Iterate reverse along target dim
        for (int64_t i = dim_size-1; i >= 0; --i) {
            sum += input[base + i * stride_dim];
            output[base + i * stride_dim] = sum;
        }
        idx += gridDim.x * blockDim.x; // stride
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    auto output = torch::empty_like(x);
    
    const int64_t dim_size = x.size(dim);
    const int64_t num_slices = x.numel() / dim_size;
    const int64_t stride_dim = x.stride(dim);
    
    const int threads = 256;
    const int blocks = (num_slices + threads - 1) / threads;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            stride_dim,
            num_slices
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumsum (CUDA)");
}