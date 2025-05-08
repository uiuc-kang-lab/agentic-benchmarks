#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride,
    const int segment_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    
    if (idx < numel / dim_size) {
        const int segment_start = (blockIdx.y * segment_size);
        const int segment_end = min(segment_start + segment_size, dim_size);
        
        scalar_t product = 1;
        if (segment_start > 0) {
            // Load the product from the previous segment
            const int64_t prev_idx = batch_idx * (stride * dim_size) + (segment_start - 1) * stride + in_idx;
            product = output[prev_idx];
        }
        
        for (int i = segment_start; i < segment_end; i++) {
            const int64_t curr_idx = batch_idx * (stride * dim_size) + i * stride + in_idx;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
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
    
    // CUDA kernel launch parameters with increased block size
    const int threads = 512;
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward (CUDA)");
}