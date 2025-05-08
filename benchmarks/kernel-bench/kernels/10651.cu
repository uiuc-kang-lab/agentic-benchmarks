#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ void compute_sequence_cumprod(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t base_offset,
    const int64_t dim_size,
    const int64_t stride) {
    scalar_t product = 1;
    for (int i = 0; i < dim_size; ++i) {
        const int64_t offset = base_offset + i * stride;
        product *= input[offset];
        output[offset] = product;
    }
}

template <typename scalar_t>
__global__ void cumprod_kernel_gridstride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t total_sequences,
    const int64_t dim_size,
    const int64_t stride) {
    
    const int64_t sequence_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int64_t s = sequence_idx; s < total_sequences; s += blockDim.x * gridDim.x) {
        const int64_t batch_idx = s / stride;
        const int64_t in_idx = s % stride;
        const int64_t base_offset = batch_idx * (dim_size * stride) + in_idx;
        
        compute_sequence_cumprod(output, input, base_offset, dim_size, stride);
    }
}

torch::Tensor cumprod_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    const auto sizes = input.sizes();
    const auto strides = input.strides();
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t numel = input.numel();
    const int64_t total_sequences = numel / dim_size;
    
    const int threads = 128;
    const int blocks = (total_sequences + threads - 1) / threads; if (blocks > 65535) blocks = 65535;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_optimized", ([&] {
        cumprod_kernel_gridstride<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_sequences,
            dim_size,
            stride
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_optimized, "Optimized cumulative product (CUDA)");
}