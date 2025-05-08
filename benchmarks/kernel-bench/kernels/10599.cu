#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_strided(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_elements) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with grid stride
    for (int64_t idx = tid; idx < total_elements; idx += grid_stride) {
        if (idx >= numel / dim_size) continue;
        
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t start_idx = batch_idx * stride * dim_size + in_idx;
        
        scalar_t product = 1;
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            const int64_t curr_idx = start_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward_strided(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t total_elements = numel / dim_size;
    
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    const int blocks = min(max_blocks, (total_elements + threads_per_block - 1) / threads_per_block);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_strided", ([&] {
        cumprod_kernel_strided<scalar_t><<<blocks, threads_per_block>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride,
            total_elements
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_strided, "Cumulative product forward strided (CUDA)");
}