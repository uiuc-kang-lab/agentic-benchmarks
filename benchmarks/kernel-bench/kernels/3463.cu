#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    float half_x = x * 0.5f; return half_x * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                          scalar_t* __restrict__ y,
                          size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better efficiency
    for(size_t i = idx; i < numel; i += stride) {
        y[i] = gelu_function<scalar_t>(x[i]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(x);
    
    const size_t total = x.numel();
    const int threads = 256;
    const int blocks = std::min(65535, int((total + threads - 1) / threads));
    
    // Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        const size_t half_size = total / 2;
        const size_t remainder = total - half_size;
        
        // Launch first half on stream1
        gelu_kernel<scalar_t><<<blocks, threads, 0, stream1>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            half_size
        );
        
        // Launch second half on stream2
        if (remainder > 0) {
            gelu_kernel<scalar_t><<<blocks, threads, 0, stream2>>>(
                x.data_ptr<scalar_t>() + half_size,
                output.data_ptr<scalar_t>() + half_size,
                remainder
            );
        }
    }));
    
    // Synchronize streams and cleanup
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with stream execution (CUDA)");
}