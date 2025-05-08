#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

template <typename scalar_t>
__global__ void gelu_kernel_unrolled(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y,
                                    const size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t unroll_factor = 4;
    
    // Process 4 elements per thread
    scalar_t vals[unroll_factor];
    scalar_t results[unroll_factor];
    
    for (size_t i = idx; i < numel; i += stride * unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            if (i + j * stride < numel) {
                vals[j] = x[i + j * stride];
                results[j] = gelu_function<scalar_t>(vals[j]);
                y[i + j * stride] = results[j];
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    
    // Optimize thread/block configuration
    const int threads = 256;
    const int blocks = std::min(65535, static_cast<int>((numel + threads * 4 - 1) / (threads * 4)));
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}