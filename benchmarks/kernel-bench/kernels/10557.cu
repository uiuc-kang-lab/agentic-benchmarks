#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_unrolled_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    // Process 4 elements per thread per iteration
    #pragma unroll 1
    for (int base_idx = tid; base_idx < total_batches; base_idx += grid_stride * 4) {
        // Process 4 separate cumulative product chains
        #pragma unroll 4
        for (int offset = 0; offset < 4; offset++) {
            const int idx = base_idx + offset * grid_stride;
            if (idx < total_batches) {
                const int batch_idx = idx / stride;
                const int in_idx = idx % stride;
                scalar_t product = 1;

                // Unroll the inner dimension loop
                #pragma unroll 4
                for (int i = 0; i < dim_size; i++) {
                    const int64_t curr_idx = batch_idx * (stride * dim_size) + i * stride + in_idx;
                    product *= input[curr_idx];
                    output[curr_idx] = product;
                }
            }
        }
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t total_batches = input.numel() / dim_size;
    
    // Optimize thread and block count for H100
    const int threads = 256;
    const int blocks = std::min(
        (total_batches + threads * 4 - 1) / (threads * 4),
        static_cast<int64_t>(256)  // Limit max blocks for better occupancy
    );
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_unrolled_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward with unrolled loops (CUDA)");
}