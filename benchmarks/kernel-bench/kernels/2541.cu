#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using shared memory
template <typename scalar_t>
__global__ void relu_kernel_shared_memory(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    extern __shared__ scalar_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // Perform ReLU in shared memory
    if (idx < size) {
        sdata[tid] = sdata[tid] > static_cast<scalar_t>(0) ? sdata[tid] : static_cast<scalar_t>(0);
    }
    __syncthreads();

    // Write back to global memory from shared memory
    if (idx < size) {
        output[idx] = sdata[tid];
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    const size_t shared_memory_size = threads * sizeof(float); // Assume float for shared memory size calculation

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_shared_memory", ([&] {
        relu_kernel_shared_memory<scalar_t><<<blocks, threads, shared_memory_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using shared memory (CUDA)");
}