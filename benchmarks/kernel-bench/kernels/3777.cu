#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Threshold constants in constant memory
__constant__ float c_upper_threshold_float = 20.0f;
__constant__ float c_lower_threshold_float = -20.0f;
__constant__ double c_upper_threshold_double = 20.0;
__constant__ double c_lower_threshold_double = -20.0;

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        if (x > c_upper_threshold_float) return x;
        if (x < c_lower_threshold_float) return __expf(x);
        return log1pf(__expf(x));
    } else {
        if (x > c_upper_threshold_double) return x;
        if (x < c_lower_threshold_double) return exp(x);
        return log1p(exp(x));
    }
}

template <typename scalar_t>
__global__ void softplus_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + tid;
    const scalar_t* input_block = input + blockIdx.x * blockDim.x * 4;
    scalar_t* output_block = output + blockIdx.x * blockDim.x * 4;

    // Use shared memory for coalesced memory access
    __shared__ scalar_t shared_input[1024];
    
    // Load data into shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (idx + i * blockDim.x < size) {
            shared_input[tid + i * blockDim.x] = input_block[tid + i * blockDim.x];
        }
    }
    __syncthreads();

    // Process data from shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (idx + i * blockDim.x < size) {
            output_block[tid + i * blockDim.x] = compute_softplus(shared_input[tid + i * blockDim.x]);
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}