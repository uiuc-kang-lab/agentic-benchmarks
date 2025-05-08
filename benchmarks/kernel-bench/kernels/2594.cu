#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with shared memory staging
template <typename scalar_t>
__global__ void relu_kernel_shared(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    // Dynamically allocated shared memory
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data from global memory into shared memory
    if (idx < size) {
        sdata[tid] = input[idx];
    } else {
        // For threads outside range, initialize to zero to avoid using uninitialized data
        sdata[tid] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    // Apply ReLU in shared memory
    if (idx < size) {
        scalar_t value = sdata[tid];
        sdata[tid] = (value > 0) ? value : static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    // Write the result from shared memory back to global memory
    if (idx < size) {
        output[idx] = sdata[tid];
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_shared", ([&] {
        relu_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA) with shared memory");
}
