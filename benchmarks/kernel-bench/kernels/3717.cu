#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized HardSigmoid using warp-level operations and shared memory
template <typename scalar_t>
__global__ void hardsigmoid_kernel_warp(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         size_t numel) {
    extern __shared__ scalar_t shared_data[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // Constants for HardSigmoid
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0 / 6.0;

    // Load data into shared memory
    if (idx < numel) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Use warp-level operations for reduction
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = shared_data[tid];
        x = (x + three) * sixth;
        x = max(scalar_t(0.0), min(scalar_t(1.0), x));
        output[i] = x;
    }
}

// PyTorch forward function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;  // 8 warps per block
    const int blocks = (numel + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda_warp", ([&] {
        hardsigmoid_kernel_warp<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) using warp-level operations");
}