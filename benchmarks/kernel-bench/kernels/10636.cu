#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to compute cumulative product for a single sequence
// This function is modular and can be reused for different data types
// and configurations

template <typename scalar_t>
__device__ void compute_cumprod(
    scalar_t* output,
    const scalar_t* input,
    const int64_t base_offset,
    const int64_t dim_size,
    const int64_t stride) {
    scalar_t product = 1;
    for (int i = 0; i < dim_size; i++) {
        const int64_t offset = base_offset + i * stride;
        product *= input[offset];
        output[offset] = product;
    }
}

// Kernel function that utilizes the device function for cumulative product

template <typename scalar_t>
__global__ void cumprod_kernel_modular(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    
    if (idx < numel / dim_size) {
        const int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;
        compute_cumprod(output, input, base_offset, dim_size, stride);
    }
}

torch::Tensor cumprod_cuda_forward_modular(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Get tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    // Calculate dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    
    // Calculate total number of elements to process
    int64_t total_threads = numel / dim_size;
    
    // CUDA kernel launch parameters
    const int threads = 512;
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_modular", ([&] {
        cumprod_kernel_modular<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_modular, "Cumulative product forward modular (CUDA)");
}