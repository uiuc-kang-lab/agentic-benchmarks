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
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                          scalar_t* __restrict__ output,
                          size_t numel) {
    constexpr scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    constexpr scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector processing only for float type and when alignment requirements are met
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Check if input is aligned for vector loads
        const size_t align_mask = sizeof(float4) - 1;
        const bool is_aligned = ((reinterpret_cast<size_t>(input) & align_mask) == 0) &&
                              ((reinterpret_cast<size_t>(output) & align_mask) == 0);
        
        if (is_aligned && numel >= 4) {
            const int vec_size = 4;
            const int vec_numel = numel / vec_size;
            
            if (tid < vec_numel) {
                const float4* input4 = reinterpret_cast<const float4*>(input);
                float4* output4 = reinterpret_cast<float4*>(output);
                
                for (int i = tid; i < vec_numel; i += stride) {
                    float4 in4 = input4[i];
                    float4 out4;
                    float* out = reinterpret_cast<float*>(&out4);
                    
                    // Process vector elements
                    out[0] = (in4.x > 0.0f) ? in4.x : alpha * (my_exp(in4.x) - 1.0f);
                    out[1] = (in4.y > 0.0f) ? in4.y : alpha * (my_exp(in4.y) - 1.0f);
                    out[2] = (in4.z > 0.0f) ? in4.z : alpha * (my_exp(in4.z) - 1.0f);
                    out[3] = (in4.w > 0.0f) ? in4.w : alpha * (my_exp(in4.w) - 1.0f);
                    
                    // Apply lambda scaling
                    out4.x *= lambda;
                    out4.y *= lambda;
                    out4.z *= lambda;
                    out4.w *= lambda;
                    
                    output4[i] = out4;
                }
            }
            
            // Handle remaining elements
            for (int i = tid + (vec_numel * vec_size); i < numel; i += stride) {
                scalar_t x = input[i];
                scalar_t value = (x > scalar_t(0)) ? x : alpha * (my_exp(x) - scalar_t(1));
                output[i] = lambda * value;
            }
        } else {
            // Non-vectorized path for unaligned data
            for (int i = tid; i < numel; i += stride) {
                scalar_t x = input[i];
                scalar_t value = (x > scalar_t(0)) ? x : alpha * (my_exp(x) - scalar_t(1));
                output[i] = lambda * value;
            }
        }
    } else {
        // Non-vectorized path for double type
        for (int i = tid; i < numel; i += stride) {
            scalar_t x = input[i];
            scalar_t value = (x > scalar_t(0)) ? x : alpha * (my_exp(x) - scalar_t(1));
            output[i] = lambda * value;
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = std::min(max_blocks, static_cast<int>((numel + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    return output;
}