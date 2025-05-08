#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Simple non-vectorized kernel for fallback types
template <typename scalar_t>
__global__ void tanh_kernel_simple(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        // For float use tanhf, for double use tanh; this is resolved at compile-time
        output[i] = (sizeof(scalar_t) == sizeof(float)) ? tanhf(input[i]) : tanh(input[i]);
    }
}

// Vectorized kernel for float using float4, with no divergent branching in the bulk.
// This kernel assumes that the number of elements is a multiple of 4.
__global__ void tanh_kernel_bulk_float(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    for (int i = idx; i < num_vec; i += stride) {
        // Unconditionally process 4 elements at a time
        float4 x = in4[i];
        float4 y;
        y.x = tanhf(x.x);
        y.y = tanhf(x.y);
        y.z = tanhf(x.z);
        y.w = tanhf(x.w);
        out4[i] = y;
    }
}

// Remainder kernel for float to handle leftover elements
__global__ void tanh_kernel_remainder_float(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int offset,
                                             int remaining) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < remaining) {
        output[offset + idx] = tanhf(input[offset + idx]);
    }
}

// Vectorized kernel for double using double2
__global__ void tanh_kernel_bulk_double(const double* __restrict__ input,
                                         double* __restrict__ output,
                                         int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const double2* in2 = reinterpret_cast<const double2*>(input);
    double2* out2 = reinterpret_cast<double2*>(output);
    for (int i = idx; i < num_vec; i += stride) {
        double2 x = in2[i];
        double2 y;
        y.x = tanh(x.x);
        y.y = tanh(x.y);
        out2[i] = y;
    }
}

// Remainder kernel for double
__global__ void tanh_kernel_remainder_double(const double* __restrict__ input,
                                               double* __restrict__ output,
                                               int offset,
                                               int remaining) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < remaining) {
        output[offset + idx] = tanh(input[offset + idx]);
    }
}


torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "no_divergence_vec_tanh", ([&] {
        // For float, use vectorized kernels with no divergence in the bulk portion.
        if (std::is_same<scalar_t, float>::value) {
            // Compute bulk size (multiple of 4) and remainder
            int remainder = size % 4;
            int bulk = size - remainder;
            int num_vec = bulk / 4;
            int blocks = (num_vec + threads - 1) / threads;
            tanh_kernel_bulk_float<<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                num_vec
            );
            if (remainder > 0) {
                // Launch a separate kernel for the remaining elements (small overhead)
                tanh_kernel_remainder_float<<<1, remainder>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    bulk,
                    remainder
                );
            }
        } else if (std::is_same<scalar_t, double>::value) {
            // For double, use vectorized kernels with double2 (2 doubles per vectorized load)
            int remainder = size % 2;
            int bulk = size - remainder;
            int num_vec = bulk / 2;
            int blocks = (num_vec + threads - 1) / threads;
            tanh_kernel_bulk_double<<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                num_vec
            );
            if (remainder > 0) {
                tanh_kernel_remainder_double<<<1, remainder>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    bulk,
                    remainder
                );
            }
        } else {
            // Fallback for any other floating point type
            int blocks = (size + threads - 1) / threads;
            tanh_kernel_simple<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with vectorized no divergence kernels (CUDA)");
}
