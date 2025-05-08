#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_warp(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % warpSize;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    
    if (idx < numel / dim_size) {
        scalar_t product = scalar_t(1);
        const int start_idx = batch_idx * stride * dim_size + in_idx;
        
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            const int curr_idx = start_idx + i * stride;
            scalar_t curr_val = input[curr_idx];
            
            // Initialize local product for this step
            scalar_t val = curr_val;
            
            // Perform warp-level cumulative product using type-safe shuffle
            #pragma unroll
            for (int offset = 1; offset < warpSize; offset *= 2) {
                scalar_t n;
                if (std::is_same<scalar_t, c10::Half>::value) {
                    n = __shfl_up_sync(0xffffffff, static_cast<__half>(val), offset);
                } else {
                    n = __shfl_up_sync(0xffffffff, val, offset);
                }
                
                if (lane_id >= offset) {
                    val *= n;
                }
            }
            
            // Update running product
            product *= curr_val;
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t total_threads = numel / dim_size;
    
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel_warp<scalar_t><<<blocks, threads>>>(
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