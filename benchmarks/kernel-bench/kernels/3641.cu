#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for HardSigmoid combining branchless operations and memory coalescing
template <typename scalar_t>
__global__ void optimized_hardsigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const size_t numel) {
    // Use vector loads/stores for better memory throughput
    using Vec4 = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_stride = stride * 4;
    const int vec_numel = numel / 4;
    
    // Constants as vectors for efficient computation
    const scalar_t add_const = static_cast<scalar_t>(3);
    const scalar_t div_const = static_cast<scalar_t>(1.0f/6.0f);
    
    // Vector processing
    for (size_t i = tid; i < vec_numel; i += stride) {
        const Vec4* in_vec = reinterpret_cast<const Vec4*>(input) + i;
        Vec4* out_vec = reinterpret_cast<Vec4*>(output) + i;
        
        Vec4 val = *in_vec;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scalar_t x = reinterpret_cast<scalar_t*>(&val)[j];
            scalar_t y = (x + add_const) * div_const;
            // Use intrinsic functions for better performance
            reinterpret_cast<scalar_t*>(&val)[j] = __saturatef(y);
        }
        *out_vec = val;
    }
    
    // Handle remaining elements
    const int rem_start = vec_numel * 4;
    for (size_t i = rem_start + tid; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + add_const) * div_const;
        output[i] = __saturatef(y);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize block size for better occupancy
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_hardsigmoid_cuda", ([&] {
        optimized_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardSigmoid activation forward (CUDA)");
}