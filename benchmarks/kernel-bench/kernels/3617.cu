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
__global__ void selu_kernel_predicated(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numel; 
         idx += blockDim.x * gridDim.x) {
        
        const scalar_t x = input[idx];
        
        // Create predicate masks (0.0 or 1.0)
        const scalar_t is_positive = static_cast<scalar_t>(x > 0);
        const scalar_t is_negative = static_cast<scalar_t>(1) - is_positive;
        
        // Compute both paths
        const scalar_t positive_path = x;
        const scalar_t exp_term = my_exp(x);
        const scalar_t negative_path = alpha * (exp_term - static_cast<scalar_t>(1));
        
        // Blend results using predication
        const scalar_t result = (positive_path * is_positive) + 
                              (negative_path * is_negative);
                              
        output[idx] = lambda * result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize launch configuration
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel_predicated<scalar_t><<<blocks, threads>>>(
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