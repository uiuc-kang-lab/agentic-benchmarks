#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    const scalar_t sqrt_two = 1.4142135623730951f; return x * 0.5f * (1.0f + erff(x / sqrt_two));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

template <typename scalar_t>
__global__ void gelu_kernel_warp(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ y,
                                size_t numel) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid >> 5;  // Warp ID
    const unsigned int lane = tid & 31;  // Lane within warp
    const unsigned int warp_count = blockDim.x >> 5;
    
    // Calculate global index for this thread
    size_t idx = (blockIdx.x * warp_count + wid) * 32 + lane;
    
    // Process elements with stride of warpSize
    while (idx < numel) {
        scalar_t val = x[idx];
        y[idx] = gelu_function<scalar_t>(val);
        idx += gridDim.x * warp_count * 32;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    const int threads_per_block = 256;  // 8 warps per block
    const int blocks = min(65535, (int)((numel + threads_per_block - 1) / threads_per_block));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel_warp<scalar_t><<<blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}