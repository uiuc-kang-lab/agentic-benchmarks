#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized cumulative product kernel using grid-stride loop to avoid branch divergence
// Each thread handles one contiguous sequence along the target dimension without divergent conditionals

template <typename scalar_t>
__global__ void __launch_bounds__(256) cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t stride,
    const int64_t dim_size,
    const int64_t total_threads) {

    // Use grid-stride loop for uniform work distribution
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_threads; idx += blockDim.x * gridDim.x) {
        // Compute the batch index and inner index uniformly
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        
        // Precompute base offset and initialize product
        const int64_t base_offset = batch_idx * (dim_size * stride) + in_idx;
        scalar_t prod = 1;
        
        // Cumulative product computation along the dimension with optimized memory access
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            const int64_t offset = base_offset + i * stride;
            prod *= input[offset];
            output[offset] = prod;
        }
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Get tensor sizes and strides
    auto sizes = input.sizes();
    auto strides = input.strides();

    // Calculate dimension properties: size along the target dim and its stride
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();

    // Calculate total number of sequences to process
    int64_t total_threads = numel / dim_size;

    // Determine CUDA launch configuration
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            stride,
            dim_size,
            total_threads
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward (CUDA)");
}
