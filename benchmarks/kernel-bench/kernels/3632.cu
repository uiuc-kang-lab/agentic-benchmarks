#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constants in constant memory for faster access
__constant__ float const_values_f[2];  // [offset, scale]
__constant__ double const_values_d[2]; // [offset, scale]

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_const(int idx) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        return const_values_f[idx];
    } else {
        return const_values_d[idx];
    }
}

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Get constants from constant memory
    const scalar_t offset = get_const<scalar_t>(0);  // 3.0
    const scalar_t scale = get_const<scalar_t>(1);   // 6.0
    
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + offset) / scale;
        y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
             : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    
    // Initialize constant memory
    float host_const_f[2] = {3.0f, 6.0f};
    double host_const_d[2] = {3.0, 6.0};
    cudaMemcpyToSymbol(const_values_f, host_const_f, sizeof(float) * 2);
    cudaMemcpyToSymbol(const_values_d, host_const_d, sizeof(double) * 2);
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}