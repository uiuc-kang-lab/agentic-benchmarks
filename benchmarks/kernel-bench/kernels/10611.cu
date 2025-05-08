#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel_shared_memory(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {

    extern __shared__ scalar_t shared_mem[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;

    if (idx < numel / dim_size) {
        scalar_t product = 1;
        int start_idx = batch_idx * stride * dim_size + in_idx;

        for (int i = 0; i < dim_size; i++) {
            int curr_idx = start_idx + i * stride;
            shared_mem[threadIdx.x] = input[curr_idx];
            __syncthreads();  // Ensure all threads have loaded their input

            product *= shared_mem[threadIdx.x];
            output[curr_idx] = product;

            __syncthreads();  // Ensure all threads have updated their output
        }
    }
}

torch::Tensor cumprod_cuda_forward_shared_memory(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t total_threads = numel / dim_size;
    
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;
    const int shared_mem_size = blockDim.x * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_shared_memory", ([&] {
        cumprod_kernel_shared_memory<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &cumprod_cuda_forward_shared_memory, "Cumulative product forward shared memory (CUDA)");
}