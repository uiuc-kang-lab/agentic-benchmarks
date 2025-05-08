#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__global__ void selu_kernel_aligned(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  const size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    const scalar_t alpha_lambda = alpha * lambda;

    // Process 4 elements per thread for better memory coalescing
    const size_t idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t stride = blockDim.x * gridDim.x * 4;

    // Use vector types for aligned loads when possible
    if (std::is_same<scalar_t, float>::value && (reinterpret_cast<uintptr_t>(input) % 16 == 0)) {
        for (size_t i = idx_base; i < numel; i += stride) {
            float4 in_vec;
            float4 out_vec;
            
            if (i + 3 < numel) {
                // Load 4 elements at once using __ldg
                const float4* in_ptr4 = reinterpret_cast<const float4*>(input + i);
                in_vec = __ldg(in_ptr4);
                
                // Process each component
                out_vec.x = (in_vec.x > 0) ? in_vec.x : alpha * (my_exp(in_vec.x) - 1.0f);
                out_vec.y = (in_vec.y > 0) ? in_vec.y : alpha * (my_exp(in_vec.y) - 1.0f);
                out_vec.z = (in_vec.z > 0) ? in_vec.z : alpha * (my_exp(in_vec.z) - 1.0f);
                out_vec.w = (in_vec.w > 0) ? in_vec.w : alpha * (my_exp(in_vec.w) - 1.0f);
                
                // Apply lambda and store
                out_vec.x *= lambda;
                out_vec.y *= lambda;
                out_vec.z *= lambda;
                out_vec.w *= lambda;
                
                // Store 4 elements at once
                *reinterpret_cast<float4*>(output + i) = out_vec;
            } else {
                // Handle remaining elements
                for (size_t j = 0; j < 4 && i + j < numel; ++j) {
                    scalar_t x = __ldg(input + i + j);
                    scalar_t result = (x > static_cast<scalar_t>(0))
                                    ? x
                                    : alpha * (my_exp(x) - static_cast<scalar_t>(1));
                    output[i + j] = lambda * result;
                }
            }
        }
    } else {
        // Fallback for double or unaligned memory
        for (size_t i = idx_base; i < numel; i += stride) {
            for (size_t j = 0; j < 4 && i + j < numel; ++j) {
                scalar_t x = __ldg(input + i + j);
                scalar_t result = (x > static_cast<scalar_t>(0))
                                ? x
                                : alpha * (my_exp(x) - static_cast<scalar_t>(1));
                output[i + j] = lambda * result;
            }
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize block size for H100
    const int threads = 256;
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel_aligned<scalar_t><<<blocks, threads>>>(
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