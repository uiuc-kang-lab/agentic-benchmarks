#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

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
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y,
                           size_t numel,
                           size_t offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + offset < numel) {
        y[idx + offset] = gelu_function<scalar_t>(x[idx + offset]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    const int num_streams = 4;
    const size_t numel = x.numel();
    const size_t elements_per_stream = (numel + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const size_t offset = i * elements_per_stream;
            const size_t elements_this_stream = std::min(elements_per_stream, 
                                                       numel - offset);
            const int blocks = (elements_this_stream + threads - 1) / threads;
            
            if (elements_this_stream > 0) {
                gelu_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    numel,
                    offset
                );
            }
        }
    }));

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}