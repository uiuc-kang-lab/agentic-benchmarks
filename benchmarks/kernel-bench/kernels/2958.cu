#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Shared memory for inter-warp coordination
    extern __shared__ scalar_t shared[];
    
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    // Global index calculation
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    
    // Process elements in chunks of 4 using vectorized loads when possible
    constexpr unsigned int vec_size = 4;
    const unsigned int vec_offset = global_idx * vec_size;
    const unsigned int vec_stride = stride * vec_size;
    
    // Main processing loop
    #pragma unroll
    for (unsigned int idx = vec_offset; idx < size - vec_size + 1; idx += vec_stride) {
        // Load 4 elements per thread
        scalar_t val[vec_size];
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            val[i] = input[idx + i];
        }
        
        // Process elements
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            val[i] = tanh(val[i]);
        }
        
        // Use warp shuffle to optimize memory access pattern
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            unsigned int mask = __activemask();
            val[i] = __shfl_sync(mask, val[i], lane_id);
        }
        
        // Store results
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            output[idx + i] = val[i];
        }
    }
    
    // Handle remaining elements
    for (unsigned int idx = global_idx + ((size / vec_size) * vec_size);
         idx < size;
         idx += stride) {
        output[idx] = tanh(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = std::min(256, (int)((input.numel() + threads - 1) / threads));
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_warp", ([&] {
        tanh_kernel_warp<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with warp synchronization (CUDA)");
}