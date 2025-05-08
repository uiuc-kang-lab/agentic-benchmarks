#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_enhanced(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    
    extern __shared__ scalar_t shared_mem[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel / dim_size) return;
    
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    const int start_idx = batch_idx * stride * dim_size + in_idx;
    
    scalar_t product = input[start_idx];
    shared_mem[threadIdx.x] = product;
    output[start_idx] = product;
    
    #pragma unroll 4
    for (int i = 1; i < dim_size; i++) {
        const int curr_idx = start_idx + i * stride;
        product *= input[curr_idx];
        shared_mem[threadIdx.x] = product;
        __syncthreads();
        
        output[curr_idx] = shared_mem[threadIdx.x];
    }
}

torch::Tensor cumprod_cuda_forward_enhanced(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t total_threads = numel / dim_size;
    
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(typename std::iterator_traits<scalar_t*>::value_type);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_enhanced", ([&] {
        cumprod_kernel_enhanced<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &cumprod_cuda_forward_enhanced, "Cumulative product forward enhanced (CUDA)");
}