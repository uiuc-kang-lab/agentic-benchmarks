#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_optimized(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t batch_count) {
    
    // 2D thread block configuration
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_count || stride_idx >= stride) return;
    
    // Calculate base index for this thread
    const int base_idx = batch_idx * (stride * dim_size) + stride_idx;
    
    // Compute cumulative product
    scalar_t product = 1;
    #pragma unroll 4
    for (int i = 0; i < dim_size; i++) {
        const int curr_idx = base_idx + i * stride;
        product *= input[curr_idx];
        output[curr_idx] = product;
    }
}

torch::Tensor cumprod_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t batch_count = numel / (dim_size * stride);
    
    // 2D block and grid configuration
    dim3 threads(16, 16);  // 256 threads total
    dim3 blocks(
        (batch_count + threads.x - 1) / threads.x,
        (stride + threads.y - 1) / threads.y
    );
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_optimized", ([&] {
        cumprod_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride,
            batch_count
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_optimized, "Cumulative product forward optimized (CUDA)");
}