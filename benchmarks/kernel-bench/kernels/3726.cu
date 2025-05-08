#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized HardSigmoid using vectorized operations and loop unrolling
template<typename scalar_t, int VEC_SIZE>
__global__ __launch_bounds__(256) void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0/6.0;
    constexpr scalar_t zero = 0.0;
    constexpr scalar_t one = 1.0;
    
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value, float4,
        typename std::conditional<std::is_same<scalar_t, double>::value, double2, void>::type
    >::type;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    const int vec_stride = blockDim.x * gridDim.x * VEC_SIZE;

    // Pre-compute constant term
    const scalar_t offset = three * sixth;

    for (int i = vec_tid; i < numel; i += vec_stride) {
        vec_t vec_in;
        scalar_t elements[VEC_SIZE];
        
        // Vectorized load
        vec_in = *reinterpret_cast<const vec_t*>(&input[i]);
        *reinterpret_cast<vec_t*>(elements) = vec_in;

        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            scalar_t x = elements[v];
            x = fma(x, sixth, offset);  // (x + 3) / 6
            x = max(zero, min(one, x));  // Use type-generic math functions
            elements[v] = x;
        }

        // Vectorized store
        *reinterpret_cast<vec_t*>(&output[i]) = *reinterpret_cast<vec_t*>(elements);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(float);  // 4 for float, 2 for double
    const int threads = 256;
    const int blocks = (numel + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t, VEC_SIZE><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}