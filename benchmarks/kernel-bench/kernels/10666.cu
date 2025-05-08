#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ void compute_cumprod_unroll(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t base_offset,
    const int64_t dim_size,
    const int64_t stride) {
    scalar_t product = 1;
    int i = 0;
    
    // Process elements in chunks of 4 with loop unrolling
    for (; i <= dim_size - 4; i += 4) {
        product *= input[base_offset + i * stride];
        output[base_offset + i * stride] = product;
        
        product *= input[base_offset + (i+1) * stride];
        output[base_offset + (i+1) * stride] = product;
        
        product *= input[base_offset + (i+2) * stride];
        output[base_offset + (i+2) * stride] = product;
        
        product *= input[base_offset + (i+3) * stride];
        output[base_offset + (i+3) * stride] = product;
    }
    
    // Process remaining elements
    for (; i < dim_size; ++i) {
        product *= input[base_offset + i * stride];
        output[base_offset + i * stride] = product;
    }
}

template <typename scalar_t>
__global__ void cumprod_kernel_vectorized(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;
    
    for (int current_idx = idx; current_idx < numel / dim_size; current_idx += grid_size) {
        const int batch_idx = current_idx / stride;
        const int in_idx = current_idx % stride;
        const int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;
        compute_cumprod_unroll(output, input, base_offset, dim_size, stride);
    }
}

torch::Tensor cumprod_cuda_forward_vectorized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t numel = input.numel();
    
    const int threads = 256;
    const int blocks = (numel / dim_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_vectorized", ([&] {
        cumprod_kernel_vectorized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward_vectorized, "Cumulative product forward vectorized (CUDA)");
}