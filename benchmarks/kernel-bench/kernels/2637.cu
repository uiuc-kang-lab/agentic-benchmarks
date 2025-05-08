#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using shared memory
template <typename scalar_t>
__global__ void relu_kernel_shared_memory(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    extern __shared__ __shared__ scalar_t shared_mem[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;

    if (idx < size) {
        // Load data into shared memory
        shared_mem[local_idx] = input[idx];
        __syncthreads();

        // Perform ReLU operation
        scalar_t val = shared_mem[local_idx];
        shared_mem[local_idx] = val > 0 ? val : 0;
        __syncthreads();

        // Write data back to global memory
        output[idx] = shared_mem[local_idx];
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_shared_memory", ([&] {
        relu_kernel_shared_memory<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
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