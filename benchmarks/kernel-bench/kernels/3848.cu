#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    if (x > 20.0) {
        return x;
    } else if (x < -20.0) {
        return exp(x);
    } else {
        return log1p(exp(x));
    }
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    extern __shared__ scalar_t shared_mem[];
    
    // Process 4 elements per thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * 4;
    const int local_tid = threadIdx.x;
    
    // Align starting index to process 4 elements at once
    int idx = tid * 4;
    
    while (idx < size - 3) {
        // Load 4 elements into shared memory with padding to avoid bank conflicts
        shared_mem[local_tid * 4] = input[idx];
        shared_mem[local_tid * 4 + 1] = input[idx + 1];
        shared_mem[local_tid * 4 + 2] = input[idx + 2];
        shared_mem[local_tid * 4 + 3] = input[idx + 3];
        
        __syncthreads();
        
        // Process 4 elements from shared memory
        scalar_t x1 = shared_mem[local_tid * 4];
        scalar_t x2 = shared_mem[local_tid * 4 + 1];
        scalar_t x3 = shared_mem[local_tid * 4 + 2];
        scalar_t x4 = shared_mem[local_tid * 4 + 3];
        
        // Compute results
        output[idx] = compute_softplus(x1);
        output[idx + 1] = compute_softplus(x2);
        output[idx + 2] = compute_softplus(x3);
        output[idx + 3] = compute_softplus(x4);
        
        __syncthreads();
        idx += stride;
    }
    
    // Handle remaining elements
    while (idx < size) {
        output[idx] = compute_softplus(input[idx]);
        idx++;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Adjust block and grid size for vectorized processing
    const int threads = 256;
    const int blocks = min(65535, (size + (threads * 4) - 1) / (threads * 4));
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}