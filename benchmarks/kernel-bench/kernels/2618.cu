#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using shared memory and warp-level primitives
template <typename scalar_t>
__global__ void relu_kernel_shared_memory(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    extern __shared__ scalar_t shared_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0;
    }
    __syncthreads();

    // Apply ReLU in shared memory
    if (idx < size) {
        shared_data[tid] = shared_data[tid] > 0 ? shared_data[tid] : 0;
    }
    __syncthreads();

    // Write back to global memory
    if (idx < size) {
        output[idx] = shared_data[tid];
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