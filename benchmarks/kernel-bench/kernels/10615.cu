#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_enhanced(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t elements_per_thread) {
    
    // Calculate thread indices
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= numel / dim_size) return;

    // Calculate batch and input indices
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    
    // Pre-calculate start index to avoid repeated computation
    const int start_idx = batch_idx * stride * dim_size + in_idx;
    
    // Shared memory for intermediate products
    extern __shared__ scalar_t shared_products[];
    
    // Process multiple elements per thread for better utilization
    for (int chunk = 0; chunk < elements_per_thread; chunk++) {
        const int thread_offset = chunk * blockDim.x;
        if (idx + thread_offset >= numel / dim_size) break;
        
        scalar_t product = 1;
        #pragma unroll 4  // Unroll small loops for better performance
        for (int i = 0; i < dim_size; i++) {
            const int curr_idx = start_idx + thread_offset * stride * dim_size + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward_enhanced(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t numel = input.numel();
    const int64_t total_threads = numel / dim_size;
    
    // Optimize thread block size and elements per thread
    const int threads = 256;  // Multiple of warp size (32)
    const int elements_per_thread = 4;  // Process multiple elements per thread
    const int blocks = (total_threads + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);
    
    // Calculate shared memory size
    const int shared_mem_size = threads * sizeof(typename std::iterator_traits<scalar_t*>::value_type);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_enhanced", ([&] {
        cumprod_kernel_enhanced<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride,
            elements_per_thread
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_enhanced, "Cumulative product forward enhanced (CUDA)");
}