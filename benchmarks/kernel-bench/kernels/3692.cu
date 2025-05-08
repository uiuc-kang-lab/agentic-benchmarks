#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store constants in constant memory for faster access
__constant__ float c_three_f = 3.0f;
__constant__ float c_inv6_f = 0.16666667f;
__constant__ double c_three_d = 3.0;
__constant__ double c_inv6_d = 0.16666666666666666;

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_three() {
    return sizeof(scalar_t) == sizeof(float) ? c_three_f : c_three_d;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_inv6() {
    return sizeof(scalar_t) == sizeof(float) ? c_inv6_f : c_inv6_d;
}

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Load constants once per thread
    const scalar_t three = get_three<scalar_t>();
    const scalar_t inv6 = get_inv6<scalar_t>();
    
    // Use vectorized loads where possible
    #pragma unroll 4
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        // Fused multiply-add operation
        scalar_t y = fma(x, inv6, three * inv6);
        // Branchless min/max using built-in functions
        y = max(scalar_t(0), min(scalar_t(1), y));
        output[i] = y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize block size based on GPU compute capability
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardSigmoid activation forward (CUDA)");
}