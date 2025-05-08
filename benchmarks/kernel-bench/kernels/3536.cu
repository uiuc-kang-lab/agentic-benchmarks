#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants for SELU activation
constexpr float ALPHA = 1.67326324235437728481f;
constexpr float LAMBDA = 1.05070098735548049342f;

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
__device__ inline void process_element(scalar_t x, scalar_t& result) {
    result = (x > scalar_t(0))
        ? x
        : static_cast<scalar_t>(ALPHA) * (my_exp(x) - scalar_t(1));
    result *= static_cast<scalar_t>(LAMBDA);
}

template <typename scalar_t>
__global__ void selu_kernel_hybrid(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
        // Vectorized processing for float
        const size_t stride = blockDim.x * gridDim.x;
        const size_t vector_stride = stride * 4;
        size_t vector_idx = idx * 4;

        // Process elements in chunks of 4
        for (; vector_idx < (numel & ~3); vector_idx += vector_stride) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[vector_idx >> 2];
            float4 out_vec;

            process_element(in_vec.x, out_vec.x);
            process_element(in_vec.y, out_vec.y);
            process_element(in_vec.z, out_vec.z);
            process_element(in_vec.w, out_vec.w);

            reinterpret_cast<float4*>(output)[vector_idx >> 2] = out_vec;
        }

        // Handle remaining elements
        const size_t remaining_start = numel & ~3;
        for (size_t i = remaining_start + idx; i < numel; i += stride) {
            process_element(input[i], output[i]);
        }
    } else {
        // Regular processing for other types
        if (idx < numel) {
            process_element(input[idx], output[idx]);
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        if (std::is_same_v<scalar_t, float>) {
            // Optimized parameters for float32
            const int threads = 256;
            const int blocks = (numel + threads * 4 - 1) / (threads * 4);
            selu_kernel_hybrid<<<blocks, threads>>>(input_ptr, output_ptr, numel);
        } else {
            // Default parameters for other types
            const int threads = 1024;
            const int blocks = (numel + threads - 1) / threads;
            selu_kernel_hybrid<<<blocks, threads>>>(input_ptr, output_ptr, numel);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (Hybrid CUDA)");
}