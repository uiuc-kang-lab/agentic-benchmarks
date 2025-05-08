#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Combined kernel: uses loop unrolling for inner loop and while loop for batch iteration

template <typename scalar_t>
__global__ void cumprod_combined_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process multiple batches per thread
    while (idx < total_batches) {
        // Decode the current batch and the starting index within the batch
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;

        scalar_t product = 1;
        int i = 0;

        // Unrolled loop: process 8 elements per iteration
        #pragma unroll 8
        for (; i + 7 < dim_size; i += 8) {
            int64_t offset = base_idx + i * stride;
            product *= input[offset];
            output[offset] = product;

            product *= input[offset + stride];
            output[offset + stride] = product;

            product *= input[offset + 2 * stride];
            output[offset + 2 * stride] = product;

            product *= input[offset + 3 * stride];
            output[offset + 3 * stride] = product;

            product *= input[offset + 4 * stride];
            output[offset + 4 * stride] = product;

            product *= input[offset + 5 * stride];
            output[offset + 5 * stride] = product;

            product *= input[offset + 6 * stride];
            output[offset + 6 * stride] = product;

            product *= input[offset + 7 * stride];
            output[offset + 7 * stride] = product;
        }

        // Process any remaining elements
        for (; i < dim_size; i++) {
            int64_t curr_idx = base_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }

        idx += blockDim.x * gridDim.x;
    }
}


// CUDA forward function

torch::Tensor cumprod_cuda_combined_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Extract tensor sizes and strides
    auto sizes = input.sizes();
    auto strides = input.strides();

    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_batches = input.numel() / dim_size;

    // Use a higher thread count per block for better occupancy
    const int threads = 512;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_combined", ([&] {
        cumprod_combined_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_combined_forward, "Combined Cumulative Product Forward (CUDA with Unroll and Stride Loop)");
}
