#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device GELU function for float and double
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) using fast math intrinsics
    // Use __fmaf_rn for fused multiply-add operations which are faster
    const float sqrt2_inv = 0.7071067811865475f;
    float cdf = 1.0f + erff(__fmaf_rn(x, sqrt2_inv, 0.0f));
    return __fmaf_rn(x, cdf, 0.0f) * 0.5f;
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * 0.7071067811865475));
}

// CUDA kernel with manual loop unrolling to reduce loop overhead
template <typename scalar_t>
__global__ void gelu_kernel_unroll_manual(const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ y,
                                           size_t numel) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Unroll the grid-stride loop by a factor of 4
    for (; tid + 3 * stride < numel; tid += 4 * stride) {
        #pragma unroll
        {
            y[tid]             = gelu_function<scalar_t>(x[tid]);
            y[tid + stride]    = gelu_function<scalar_t>(x[tid + stride]);
            y[tid + 2 * stride]= gelu_function<scalar_t>(x[tid + 2 * stride]);
            y[tid + 3 * stride]= gelu_function<scalar_t>(x[tid + 3 * stride]);
        }
    }
    // Process any remaining elements
    for (; tid < numel; tid += stride) {
        y[tid] = gelu_function<scalar_t>(x[tid]);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_unroll_manual_cuda", ([&] {
        gelu_kernel_unroll_manual<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                                 output.data_ptr<scalar_t>(),
                                                                 numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward with manual loop unrolling (CUDA)");
}
