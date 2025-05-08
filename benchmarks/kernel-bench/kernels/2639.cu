#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_shared(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = block_size * gridDim.x;
    const int block_offset = blockIdx.x * block_size;
    
    // Process multiple chunks per block
    for (int chunk_start = block_offset; chunk_start < size; chunk_start += grid_size) {
        const int chunk_idx = chunk_start + tid;
        
        // Load into shared memory
        if (chunk_idx < size) {
            shared_data[tid] = input[chunk_idx];
        }
        __syncthreads();
        
        // Process data in shared memory
        if (chunk_idx < size) {
            // Avoid re-reading from shared memory by using the already loaded value
            shared_data[tid] = max(shared_data[tid], scalar_t(0));
        }
        __syncthreads();
        
        // Write back to global memory
        if (chunk_idx < size) {
            output[chunk_idx] = shared_data[tid];
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(65535, int((input.numel() + threads - 1) / threads));
    const int shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_shared", ([&] {
        relu_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with shared memory (CUDA)");
}