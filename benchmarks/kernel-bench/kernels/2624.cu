#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using shared memory for ReLU activation
template <typename scalar_t>
__global__ void relu_shared_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    extern __shared__ __shared__ scalar_t sdata[blockDim.x];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    // Load data from global memory into shared memory if index is valid
    if (idx < size) {
        sdata[tid] = input[idx];
    }

    // Synchronize to ensure the shared memory is fully populated
    __syncthreads();

    // Compute ReLU from shared memory and write the result back to global memory
    if (idx < size) {
        scalar_t val = sdata[tid];
        output[idx] = val > 0 ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_shared_kernel", ([&] {
        relu_shared_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
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
