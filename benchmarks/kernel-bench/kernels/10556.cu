#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_shared_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx < total_batches) {
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        
        // Load first element into shared memory
        shared_data[tid] = input[base_idx];
        scalar_t product = shared_data[tid];
        output[base_idx] = product;
        
        // Process remaining elements using shared memory
        for (int i = 1; i < dim_size; i++) {
            const int64_t curr_idx = base_idx + i * stride;
            shared_data[tid] = input[curr_idx];
            __syncthreads();
            
            product *= shared_data[tid];
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
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &cumprod_cuda_forward, "Cumulative product forward with shared memory (CUDA)");
}