#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // Use fast math intrinsics for better performance
    const float inv_sqrt_2 = 0.707106781186547524f;
    float cdf = 0.5f * (1.0f + __erff(x * inv_sqrt_2));
    return __fmaf_rn(x, cdf, 0.0f);  // Fused multiply-add for better performance
}

template <>
__device__ inline double gelu_function<double>(double x) {
    const double inv_sqrt_2 = 0.707106781186547524;
    return x * 0.5 * (1.0 + erf(x * inv_sqrt_2));
}

template <typename scalar_t>
__global__ void gelu_kernel_2d(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int rows,
    const int cols) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        y[idx] = gelu_function<scalar_t>(x[idx]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() <= 2, "Input tensor must be 1D or 2D");
    
    auto output = torch::empty_like(x);
    
    // Handle both 1D and 2D cases
    const int rows = x.dim() == 2 ? x.size(0) : 1;
    const int cols = x.dim() == 2 ? x.size(1) : x.size(0);
    
    // Use 32x32 thread blocks
    const dim3 threads(32, 32);
    const dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel_2d<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}