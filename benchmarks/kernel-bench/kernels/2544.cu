#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Kernel specialized for float using vectorized loads with float4
__global__ void relu_kernel_vectorized_float(float* __restrict__ output,
                                                const float* __restrict__ input,
                                                const int64_t size) {
    const int total_threads = blockDim.x * gridDim.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_vec = size / 4;  // number of vectorized groups of 4 floats

    // Process the bulk of data using vectorized loads
    float4* out_vec = reinterpret_cast<float4*>(output);
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    for (int i = idx; i < n_vec; i += total_threads) {
        float4 data = in_vec[i];
        // Manual unrolling: process each of the 4 elements
        data.x = data.x > 0.f ? data.x : 0.f;
        data.y = data.y > 0.f ? data.y : 0.f;
        data.z = data.z > 0.f ? data.z : 0.f;
        data.w = data.w > 0.f ? data.w : 0.f;
        out_vec[i] = data;
    }

    // Handle any remaining elements
    int start = n_vec * 4;
    for (int i = start + idx; i < size; i += total_threads) {
        float val = input[i];
        output[i] = val > 0.f ? val : 0.f;
    }
}

// Kernel specialized for double using vectorized loads with double2
__global__ void relu_kernel_vectorized_double(double* __restrict__ output,
                                                 const double* __restrict__ input,
                                                 const int64_t size) {
    const int total_threads = blockDim.x * gridDim.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_vec = size / 2;  // each double2 handles 2 doubles

    double2* out_vec = reinterpret_cast<double2*>(output);
    const double2* in_vec = reinterpret_cast<const double2*>(input);
    for (int i = idx; i < n_vec; i += total_threads) {
        double2 data = in_vec[i];
        data.x = data.x > 0.0 ? data.x : 0.0;
        data.y = data.y > 0.0 ? data.y : 0.0;
        out_vec[i] = data;
    }

    int start = n_vec * 2;
    for (int i = start + idx; i < size; i += total_threads) {
        double val = input[i];
        output[i] = val > 0.0 ? val : 0.0;
    }
}

// Generic kernel with manual loop unrolling for other scalar types
template <typename scalar_t>
__global__ void relu_kernel_vectorized_generic(scalar_t* __restrict__ output,
                                                  const scalar_t* __restrict__ input,
                                                  const int64_t size) {
    const int total_threads = blockDim.x * gridDim.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Process 4 elements per iteration to reduce loop overhead
    for (int i = idx; i < size; i += total_threads * 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int index = i + j * total_threads;
            if (index < size) {
                scalar_t val = input[index];
                output[index] = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
            }
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_vectorized", ([&] {
        // Use specialized vectorized kernels for float and double
        if (std::is_same<scalar_t, float>::value) {
            relu_kernel_vectorized_float<<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.numel());
        } else if (std::is_same<scalar_t, double>::value) {
            relu_kernel_vectorized_double<<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.numel());
        } else {
            // Fallback for other types
            relu_kernel_vectorized_generic<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.numel());
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with vectorized unroll (CUDA)");
}
