#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for frequently used values
__constant__ float d_constants[2];  // [0] = 3.0f, [1] = 1.0f/6.0f
__constant__ double d_constants_double[2];  // [0] = 3.0, [1] = 1.0/6.0

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Select appropriate constants based on data type
    const scalar_t add_const = (sizeof(scalar_t) == sizeof(float)) ? 
        d_constants[0] : d_constants_double[0];
    const scalar_t mul_const = (sizeof(scalar_t) == sizeof(float)) ? 
        d_constants[1] : d_constants_double[1];
    
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + add_const) * mul_const;
        y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
             : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    // Initialize constant memory only once
    static bool constants_initialized = false;
    if (!constants_initialized) {
        float h_constants[2] = {3.0f, 1.0f/6.0f};
        double h_constants_double[2] = {3.0, 1.0/6.0};
        cudaMemcpyToSymbol(d_constants, h_constants, sizeof(h_constants));
        cudaMemcpyToSymbol(d_constants_double, h_constants_double, sizeof(h_constants_double));
        constants_initialized = true;
    }

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