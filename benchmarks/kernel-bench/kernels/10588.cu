#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;
    const int tid = threadIdx.x;
    
    if (idx < numel / dim_size) {
        scalar_t product = 1;
        
        // Process in chunks to maximize shared memory usage
        const int CHUNK_SIZE = 16;
        for (int chunk = 0; chunk < dim_size; chunk += CHUNK_SIZE) {
            const int chunk_end = min(chunk + CHUNK_SIZE, dim_size);
            
            // Load chunk into shared memory
            for (int i = chunk; i < chunk_end; i++) {
                const int64_t curr_idx = batch_idx * (stride * dim_size) + i * stride + in_idx;
                shared_data[tid * CHUNK_SIZE + (i - chunk)] = input[curr_idx];
            }
            __syncthreads();
            
            // Process chunk
            for (int i = chunk; i < chunk_end; i++) {
                product *= shared_data[tid * CHUNK_SIZE + (i - chunk)];
                const int64_t curr_idx = batch_idx * (stride * dim_size) + i * stride + in_idx;
                output[curr_idx] = product;
            }
            __syncthreads();
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
    const int shared_mem_size = threads * 16 * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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