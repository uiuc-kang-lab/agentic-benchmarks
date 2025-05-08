#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel with stride loop to cover workloads larger than available threads

template <typename scalar_t>
__global__ void cumprod_stride_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    // Use stride looping to cover all batch indices
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        scalar_t product = 1;
        
        // Perform cumulative product along the dimension
        for (int i = 0; i < dim_size; i++) {
            int64_t curr_idx = batch_idx * (dim_size * stride) + i * stride + in_idx;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    // Create an empty output tensor with the same size as input
    auto output = torch::empty_like(input);
    
    // Retrieve tensor size and stride information
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    // Dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    
    // Total number of cumulative-product batches to process
    int64_t total_batches = input.numel() / dim_size;

    // Set CUDA kernel launch parameters
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_stride_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward (CUDA with stride loop)");
}
