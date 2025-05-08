#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants in constant memory for faster access
__constant__ float d_alpha = 1.67326324235437728481f;
__constant__ float d_lambda = 1.05070098735548049342f;
__constant__ double d_alpha_double = 1.67326324235437728481;
__constant__ double d_lambda_double = 1.05070098735548049342;

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
__global__ void selu_kernel_uniform(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  const size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    const scalar_t x = input[idx];
    const scalar_t alpha = sizeof(scalar_t) == sizeof(float) ? d_alpha : d_alpha_double;
    const scalar_t lambda = sizeof(scalar_t) == sizeof(float) ? d_lambda : d_lambda_double;
    
    // Predicated execution using multiplication instead of branches
    const scalar_t pos_mask = __int2float_rn(x > 0);
    const scalar_t neg_mask = scalar_t(1) - pos_mask;
    
    // Compute both paths and blend results using masks
    const scalar_t pos_result = x;
    const scalar_t neg_result = alpha * (my_exp(x) - scalar_t(1));
    
    // Blend results using predication
    const scalar_t result = pos_mask * pos_result + neg_mask * neg_result;
    output[idx] = lambda * result;
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize thread/block configuration
    const int threads = 256;  // Reduced thread count for better occupancy
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel_uniform<scalar_t><<<blocks, threads>>>(
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