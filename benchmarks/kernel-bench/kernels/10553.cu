#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel that precomputes the base offset and unrolls the inner loop
template <typename scalar_t>
__global__ void cumprod_optimized_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    // Grid-stride loop to cover all batches
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        
        // Determine the batch and the input index within the stride
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        
        // Precompute the base pointer offset for the current series
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        scalar_t product = static_cast<scalar_t>(1);

        int i = 0;
        // Unroll loop in groups of 4 to reduce loop overhead
        #pragma unroll 4
        for(; i <= dim_size - 4; i += 4) {
            const int64_t idx0 = base_idx + i * stride;
            product *= input[idx0];
            output[idx0] = product;
            
            const int64_t idx1 = base_idx + (i + 1) * stride;
            product *= input[idx1];
            output[idx1] = product;
            
            const int64_t idx2 = base_idx + (i + 2) * stride;
            product *= input[idx2];
            output[idx2] = product;
            
            const int64_t idx3 = base_idx + (i + 3) * stride;
            product *= input[idx3];
            output[idx3] = product;
        }

        // Handle any remaining elements
        for(; i < dim_size; i++) {
            const int64_t cur_idx = base_idx + i * stride;
            product *= input[cur_idx];
            output[cur_idx] = product;
        }
    }
}

// CUDA forward function
torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Retrieve tensor dimensions and strides
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t total_batches = input.numel() / dim_size;
    
    // Set kernel launch parameters
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_optimized_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward (optimized CUDA)");
}
