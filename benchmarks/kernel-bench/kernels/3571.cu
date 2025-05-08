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
    
    using Vec4 = typename std::conditional<std::is_same<scalar_t, float>::value,
                                         float4, double4>::type;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    const int vec_numel = numel / vec_size;
    
    if (std::is_same<scalar_t, float>::value && tid < vec_numel) {
        const Vec4* input4 = reinterpret_cast<const Vec4*>(input);
        Vec4* output4 = reinterpret_cast<Vec4*>(output);
        
        for (int i = tid; i < vec_numel; i += stride) {
            Vec4 in4 = input4[i];
            Vec4 out4;
            
            scalar_t* in = reinterpret_cast<scalar_t*>(&in4);
            scalar_t* out = reinterpret_cast<scalar_t*>(&out4);
            
            #pragma unroll
            for (int j = 0; j < vec_size; j++) {
                scalar_t x = in[j];
                scalar_t value = (x > scalar_t(0)) ? 
                    x : alpha * (my_exp(x) - scalar_t(1));
                out[j] = lambda * value;
            }
            
            output4[i] = out4;
        }
    }
    
    for (int i = tid + (vec_numel * vec_size); i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t value = (x > scalar_t(0)) ? 
            x : alpha * (my_exp(x) - scalar_t(1));
        output[i] = lambda * value;
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