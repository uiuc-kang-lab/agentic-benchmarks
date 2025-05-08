#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_warp_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    
    const int64_t grid_size = blockDim.x * gridDim.x;
    const int64_t batch_total = numel / dim_size;
    
    for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         linear_idx < batch_total; 
         linear_idx += grid_size) {
        const int64_t batch_idx = linear_idx / stride;
        const int64_t in_idx = linear_idx % stride;
        const int64_t base_offset = batch_idx * dim_size * stride + in_idx;
        
        scalar_t product = 1;
        for (int64_t i = 0; i < dim_size; ++i) {
            const int64_t offset = base_offset + i * stride;
            product *= input[offset];
            output[offset] = product;
        }
    }
}

torch::Tensor cumprod_cuda_warp_optimized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t numel = input.numel();
    
    const int threads = 512; // 512 is a multiple of warp size (32)
    const int blocks = (numel / dim_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_warp_opt", ([&] {
        cumprod_kernel_warp_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_warp_optimized, "Cumprod warp optimized (CUDA)");
}