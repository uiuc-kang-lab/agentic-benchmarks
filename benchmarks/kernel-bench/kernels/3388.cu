#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Explicit specializations of gelu_function for float and double.
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

// Vectorized CUDA kernel that processes 4 elements per thread
template <typename scalar_t>
__global__ void gelu_kernel_vectorized(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ y,
                                      size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes 4 elements
    for (int i = tid * 4; i < numel; i += stride * 4) {
        scalar_t4 inputs;
        scalar_t4 outputs;
        
        // Load 4 elements if available
        if (i + 3 < numel) {
            inputs = *reinterpret_cast<const scalar_t4*>(x + i);
            outputs.x = gelu_function(inputs.x);
            outputs.y = gelu_function(inputs.y);
            outputs.z = gelu_function(inputs.z);
            outputs.w = gelu_function(inputs.w);
            *reinterpret_cast<scalar_t4*>(y + i) = outputs;
        } else {
            // Handle remaining elements
            for (int j = 0; j < 4 && i + j < numel; j++) {
                y[i + j] = gelu_function(x[i + j]);
            }
        }
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel_vectorized<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
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