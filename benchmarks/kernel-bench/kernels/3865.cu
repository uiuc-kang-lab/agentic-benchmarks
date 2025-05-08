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
    
    __shared__ scalar_t smem[1024]; // 256 threads * 4 elements
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int elements_per_block = blockDim.x * 4;
    int offset = bid * elements_per_block;

    // Load and process 4 elements per thread, minimize shared memory access
    scalar_t thread_data[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int load_idx = offset + tid + i * blockDim.x;
        if (load_idx < size) {
            // Load directly from global memory and process
            thread_data[i] = compute_softplus(__ldg(&input[load_idx]));
        }
    }

    // Store processed results to shared memory with padding to avoid bank conflicts
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int sm_idx = tid + i * (blockDim.x + 1); // Add padding
        if (offset + tid + i * blockDim.x < size) {
            smem[sm_idx] = thread_data[i];
        }
    }
    
    __syncthreads();

    // Store results to global memory
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int store_idx = offset + tid + i * blockDim.x;
        if (store_idx < size) {
            output[store_idx] = smem[tid + i * (blockDim.x + 1)]; // Account for padding
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (size + elements_per_block - 1) / elements_per_block;

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