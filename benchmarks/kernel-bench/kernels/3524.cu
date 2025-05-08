#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
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
__device__ __forceinline__ void process_element(const scalar_t x, scalar_t& result) {
    result = (x > static_cast<scalar_t>(0))
        ? x
        : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
    result *= static_cast<scalar_t>(1.05070098735548049342);
}

template <typename scalar_t>
__global__ void selu_kernel_aggressive_unroll(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             const size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 8;
    
    // Main loop with aggressive unrolling
    #pragma unroll
    for (size_t i = tid; i < numel - (unroll_factor - 1); i += stride * unroll_factor) {
        if (i + (unroll_factor - 1) < numel) {
            scalar_t x[unroll_factor];
            scalar_t results[unroll_factor];
            
            // Load unroll_factor elements
            #pragma unroll
            for (int j = 0; j < unroll_factor; j++) {
                x[j] = __ldg(&input[i + j * stride]);
            }
            
            // Process unroll_factor elements
            #pragma unroll
            for (int j = 0; j < unroll_factor; j++) {
                process_element(x[j], results[j]);
            }
            
            // Store unroll_factor elements
            #pragma unroll
            for (int j = 0; j < unroll_factor; j++) {
                output[i + j * stride] = results[j];
            }
        }
    }
    
    // Handle remaining elements
    for (size_t i = tid + ((numel / (stride * unroll_factor)) * stride * unroll_factor);
         i < numel;
         i += stride) {
        if (i < numel) {
            scalar_t x = __ldg(&input[i]);
            scalar_t result;
            process_element(x, result);
            output[i] = result;
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize thread configuration for H100
    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_aggressive_unroll<scalar_t><<<blocks, threads>>>(
            input_ptr, output_ptr, numel);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Aggressive Unrolling (CUDA)");
}