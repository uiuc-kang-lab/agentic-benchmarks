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
__global__ void selu_kernel_hybrid(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 size_t numel) {
    size_t total_threads = blockDim.x * gridDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate balanced work distribution
    size_t work_per_thread = (numel + total_threads - 1) / total_threads;
    size_t start = tid * work_per_thread;
    size_t end = min(start + work_per_thread, numel);
    
    // Constants for SELU
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t scale = static_cast<scalar_t>(1.05070098735548049342);
    
    // Process elements in chunks of 4 using loop unrolling
    size_t i = start;
    for (; i + 4 <= end; i += 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scalar_t x = __ldg(&input[i + j]);
            scalar_t result = (x > 0) ? x : alpha * (my_exp(x) - 1);
            output[i + j] = scale * result;
        }
    }
    
    // Handle remaining elements
    for (; i < end; i++) {
        scalar_t x = __ldg(&input[i]);
        scalar_t result = (x > 0) ? x : alpha * (my_exp(x) - 1);
        output[i] = scale * result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    const int threads = 256;  // Reduced thread count for better occupancy
    const int blocks = min(65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_hybrid_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_hybrid<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA) with Hybrid Optimization");
}