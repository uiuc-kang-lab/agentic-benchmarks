#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return __expf(x);  // Using faster CUDA intrinsic
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__global__ void selu_kernel_hybrid(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    
    // Process elements in chunks of 4 when possible
    const int aligned_size = (numel / 4) * 4;
    
    #pragma unroll 4
    for (int i = tid; i < aligned_size; i += stride * 4) {
        scalar_t x = __ldg(&input[i]);  // Using __ldg for cached load
        scalar_t result = (x > scalar_t(0)) ? 
                         x : 
                         alpha * (my_exp(x) - scalar_t(1));
        output[i] = lambda * result;
    }
    
    // Handle remaining elements
    for (int i = tid + aligned_size; i < numel; i += stride) {
        scalar_t x = __ldg(&input[i]);
        scalar_t result = (x > scalar_t(0)) ? 
                         x : 
                         alpha * (my_exp(x) - scalar_t(1));
        output[i] = lambda * result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;  // Optimized thread count
    const int blocks = min(65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel_hybrid<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA)");
}