#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Constants for computation
    const scalar_t three = 3.0f;
    const scalar_t inv6 = 1.0f/6.0f;
    
    for (size_t i = idx; i < numel; i += stride) {
        // Load data into shared memory
        shared_data[tid] = input[i];
        __syncthreads();  // Single sync point after load
        
        // Process data from shared memory
        scalar_t x = shared_data[tid];
        scalar_t y = (x + three) * inv6;
        y = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
        
        // Write directly to global memory - no sync needed
        output[i] = y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        const int shared_mem_size = threads * sizeof(scalar_t);
        hardsigmoid_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}