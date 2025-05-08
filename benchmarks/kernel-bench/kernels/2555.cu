#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using shared memory for coalesced data access.
template <typename scalar_t>
__global__ void relu_kernel_shared_memory(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    extern __shared__ scalar_t shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local_idx = threadIdx.x;

    // Load data into shared memory to ensure coalesced memory accesses
    if (idx < size) {
        shared_mem[local_idx] = input[idx];
    }
    
    // Synchronize to ensure all threads have loaded data into shared memory
    __syncthreads();

    // Perform ReLU in shared memory
    if (idx < size) {
        shared_mem[local_idx] = shared_mem[local_idx] > 0 ? shared_mem[local_idx] : 0;
    }

    // Synchronize to ensure all threads have completed computation in shared memory
    __syncthreads();

    // Store results from shared memory back to global memory
    if (idx < size) {
        output[idx] = shared_mem[local_idx];
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_shared_memory", ([&] {
        int sharedMemSize = threads * sizeof(scalar_t);
        relu_kernel_shared_memory<scalar_t><<<blocks, threads, sharedMemSize>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ReLU forward using shared memory (CUDA)");
}
