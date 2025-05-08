#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Declare constant memory for SELU parameters
__constant__ float d_alpha_f = 1.67326324235437728481f;
__constant__ float d_lambda_f = 1.05070098735548049342f;
__constant__ double d_alpha_d = 1.67326324235437728481;
__constant__ double d_lambda_d = 1.05070098735548049342;

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return __expf(x); // Use faster intrinsic
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_alpha() {
    return sizeof(scalar_t) == sizeof(float) ? d_alpha_f : d_alpha_d;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_lambda() {
    return sizeof(scalar_t) == sizeof(float) ? d_lambda_f : d_lambda_d;
}

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                           scalar_t* __restrict__ output,
                           const size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Load constants once per thread
    const scalar_t alpha = get_alpha<scalar_t>();
    const scalar_t lambda = get_lambda<scalar_t>();
    
    // Grid-stride loop for better occupancy
    #pragma unroll 4
    for (size_t i = idx; i < numel; i += stride) {
        const scalar_t x = input[i];
        // Use fused multiply-add operations where possible
        output[i] = x > scalar_t(0) ? 
            lambda * x : 
            fma(lambda * alpha, my_exp(x) - scalar_t(1), scalar_t(0));
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize thread and block count
    const int threads = 256; // Optimal for modern GPUs
    const int max_blocks = 65535;
    const int blocks = std::min(max_blocks, int((numel + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel<scalar_t><<<blocks, threads>>>(
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