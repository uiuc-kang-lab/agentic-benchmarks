#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_unroll8_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        
        scalar_t product = 1;
        int i = 0;
        
        // Main loop with unroll factor of 8
        #pragma unroll 8
        for (; i + 7 < dim_size; i += 8) {
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
            
            const int64_t idx4 = base_idx + (i + 4) * stride;
            product *= input[idx4];
            output[idx4] = product;
            
            const int64_t idx5 = base_idx + (i + 5) * stride;
            product *= input[idx5];
            output[idx5] = product;
            
            const int64_t idx6 = base_idx + (i + 6) * stride;
            product *= input[idx6];
            output[idx6] = product;
            
            const int64_t idx7 = base_idx + (i + 7) * stride;
            product *= input[idx7];
            output[idx7] = product;
        }
        
        // Handle remaining elements
        for (; i < dim_size; i++) {
            const int64_t curr_idx = base_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
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
    
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_unroll8_kernel<scalar_t><<<blocks, threads>>>(
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