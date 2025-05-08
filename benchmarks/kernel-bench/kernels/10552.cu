#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel with stride loop and boundary handling

template <typename scalar_t>
__global__ void cumprod_optimized_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use loop to handle elements with stride, check bounds
    while (idx < total_batches) {
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        scalar_t product = 1;
        
        // Compute cumulative product along dimension
        for (int i = 0; i < dim_size; ++i) {
            int64_t curr_idx = batch_idx * (dim_size * stride) + i * stride + in_idx;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }

        idx += blockDim.x * gridDim.x;
    }
}

torch::Tensor cumprod_cuda_optimized_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    // Dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_batches = input.numel() / dim_size;

    // CUDA kernel configuration
    const int threads = 512;  // Increased threads per block
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_optimized", ([&] {
        cumprod_optimized_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_optimized_forward, "Optimized Cumulative Product Forward (CUDA with Stride Loop)");
}