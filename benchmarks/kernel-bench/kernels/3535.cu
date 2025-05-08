#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return __expf(x);  // Use faster CUDA intrinsic
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// Constants for SELU activation
constexpr float ALPHA = 1.67326324235437728481f;
constexpr float SCALE = 1.05070098735548049342f;

template <typename scalar_t>
__global__ void selu_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const size_t numel
) {
    // Calculate work distribution
    const size_t total_threads = blockDim.x * gridDim.x;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = total_threads;
    
    // Use vectorized loads where possible for better memory throughput
    using Vec4 = float4;
    const Vec4* input4 = reinterpret_cast<const Vec4*>(input);
    Vec4* output4 = reinterpret_cast<Vec4*>(output);
    const size_t vec_elements = numel / 4;
    
    // Vector processing
    #pragma unroll
    for (size_t i = tid; i < vec_elements; i += stride) {
        Vec4 in4 = input4[i];
        float* in = reinterpret_cast<float*>(&in4);
        float out[4];
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = in[j];
            out[j] = SCALE * (x > 0 ? x : ALPHA * (__expf(x) - 1.0f));
        }
        
        output4[i] = *reinterpret_cast<Vec4*>(out);
    }
    
    // Handle remaining elements
    const size_t vec_offset = vec_elements * 4;
    for (size_t i = vec_offset + tid; i < numel; i += stride) {
        scalar_t x = input[i];
        output[i] = SCALE * (x > 0 ? x : ALPHA * (my_exp(x) - 1));
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize launch configuration
    const int threads = 256;  // Reduced thread count for better occupancy
    const int max_blocks = 65535;
    const int blocks = std::min(max_blocks, static_cast<int>((numel + threads - 1) / threads));
    
    // Use CUDA stream for asynchronous execution
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_optimized_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_optimized<scalar_t><<<blocks, threads, 0, stream>>>(
            input_ptr, output_ptr, numel
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Optimized SELU Activation Forward (CUDA)");
}