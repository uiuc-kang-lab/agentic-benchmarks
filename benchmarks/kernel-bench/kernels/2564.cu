#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void shared_mem_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int blockSize = blockDim.x;
    const int gridSize = blockSize * gridDim.x;
    const int local_size = min(blockSize, static_cast<int>(size - bid * blockSize));
    
    // Process multiple chunks per block
    for (int base = bid * blockSize; base < size; base += gridSize) {
        const int chunk_size = min(blockSize, static_cast<int>(size - base));
        
        // Coalesced load into shared memory
        if (tid < chunk_size) {
            shared_data[tid] = input[base + tid];
        }
        
        // Single sync point to ensure shared memory is loaded
        __syncthreads();
        
        // Process data in shared memory
        if (tid < chunk_size) {
            shared_data[tid] = shared_data[tid] > 0 ? shared_data[tid] : 0;
            // Write back to global memory
            output[base + tid] = shared_data[tid];
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_mem_relu_kernel", ([&] {
        shared_mem_relu_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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