#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized CUDA kernel with overlapped memory transfers via CUDA streams
__global__ void gelu_kernel_vectorized_streams(const float4* __restrict__ input,
                                               float4* __restrict__ output,
                                               size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in_val = __ldg(&input[idx]);
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output[idx] = in_val;
    }
}

// Kernel for the remaining elements not divisible by float4
__global__ void gelu_kernel_remainder_streams(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              size_t offset,
                                              size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pos = idx + offset;
    if (pos < numel) {
        float val = __ldg(&input[pos]);
        output[pos] = gelu_function(val);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for the streamed version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const size_t vec_size = 4; // float4 => 4 floats (128 bits)
    size_t n4 = numel / vec_size;
    size_t remainder = numel % vec_size;

    int threads = 256;
    int blocks_main = (n4 + threads - 1) / threads;
    int blocks_rem = (remainder + threads - 1) / threads;

    cudaStream_t stream_main, stream_rem;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_rem);

    if (n4 > 0) {
        gelu_kernel_vectorized_streams<<<blocks_main, threads, 0, stream_main>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }

    if (remainder > 0) {
        gelu_kernel_remainder_streams<<<blocks_rem, threads, 0, stream_rem>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            n4 * vec_size,
            numel);
    }

    cudaStreamSynchronize(stream_main);
    cudaStreamSynchronize(stream_rem);
    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_rem);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with streams");
}
