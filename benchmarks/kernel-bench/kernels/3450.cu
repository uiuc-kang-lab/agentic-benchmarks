#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// GELU function specializations
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f)); // Pre-computed 1/sqrt(2)
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * 0.70710678118654752440)); // Pre-computed 1/sqrt(2)
}

// Vectorized load/store for better memory bandwidth utilization
template <typename scalar_t>
__device__ inline void vector_load(const scalar_t* in, scalar_t& v1, scalar_t& v2, 
                                 scalar_t& v3, scalar_t& v4, size_t idx) {
    const scalar_t4* in4 = reinterpret_cast<const scalar_t4*>(in + idx);
    scalar_t4 tmp = *in4;
    v1 = tmp.x; v2 = tmp.y; v3 = tmp.z; v4 = tmp.w;
}

template <typename scalar_t>
__device__ inline void vector_store(scalar_t* out, scalar_t v1, scalar_t v2, 
                                  scalar_t v3, scalar_t v4, size_t idx) {
    scalar_t4* out4 = reinterpret_cast<scalar_t4*>(out + idx);
    *out4 = make_scalar_t4(v1, v2, v3, v4);
}

// Kernel that processes 4 elements per thread using grid-stride loop
template <typename scalar_t>
__global__ void gelu_kernel_vectorized(const scalar_t* __restrict__ in,
                                     scalar_t* __restrict__ out,
                                     size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t vector_stride = stride * 4;
    
    // Process 4 elements per thread
    for (size_t i = idx * 4; i < numel; i += vector_stride) {
        scalar_t v1, v2, v3, v4;
        
        if (i + 3 < numel) {
            vector_load(in, v1, v2, v3, v4, i);
            v1 = gelu_function(v1);
            v2 = gelu_function(v2);
            v3 = gelu_function(v3);
            v4 = gelu_function(v4);
            vector_store(out, v1, v2, v3, v4, i);
        } else {
            // Handle boundary case
            for (size_t j = 0; j < 4 && i + j < numel; j++) {
                out[i + j] = gelu_function(in[i + j]);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    
    // Choose optimal launch configuration
    const int threads = 256;
    const int max_blocks = 1024;
    const int blocks = std::min(max_blocks, int((numel + threads * 4 - 1) / (threads * 4)));
    
    // For small tensors, use simple kernel
    if (numel < 1024) {
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_small", ([&] {
            gelu_kernel_vectorized<scalar_t><<<1, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }));
    } else {
        // Use stream for async execution
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_large", ([&] {
            gelu_kernel_vectorized<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }));
        
        cudaStreamDestroy(stream);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized GELU activation forward (CUDA)");
}