#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Vectorized kernel for float using float4 with uniform control flow
__global__ void vectorized_sigmoid_kernel_float(const float* __restrict__ input,
                                                  float* __restrict__ output,
                                                  const int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // Each thread processes a block of consecutive float4 groups
    for (int i = idx; i < n_vec; i += stride) {
        // Load a group of 4 floats at once
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        // Compute sigmoid uniformly for each component
        out_val.x = 1.0f / (1.0f + expf(-in_val.x));
        out_val.y = 1.0f / (1.0f + expf(-in_val.y));
        out_val.z = 1.0f / (1.0f + expf(-in_val.z));
        out_val.w = 1.0f / (1.0f + expf(-in_val.w));
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
}

// Tail kernel for float to process leftover elements with uniform thread count
__global__ void tail_sigmoid_kernel_float(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int start,
                                            const int tail_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Launch exactly 'tail_size' threads to minimize divergence in the tail
    if (idx < tail_size) {
        int i = start + idx;
        float in_val = input[i];
        float out_val = 1.0f / (1.0f + expf(-in_val));
        output[i] = out_val;
    }
}

// Fallback scalar kernel for non-float types
template <typename scalar_t>
__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        float in_val = static_cast<float>(input[i]);
        float out_val = 1.0f / (1.0f + expf(-in_val));
        output[i] = static_cast<scalar_t>(out_val);
    }
}

// Forward function dispatches to vectorized kernels for float and a scalar kernel for others
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;

    if (input.scalar_type() == at::ScalarType::Float) {
        // Compute the number of groups of 4 elements
        int n_vec = size / 4;
        int tail = size - (n_vec * 4);
        
        // Launch vectorized kernel if there is a complete float4 block
        if (n_vec > 0) {
            int blocks = (n_vec + threads - 1) / threads;
            vectorized_sigmoid_kernel_float<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                n_vec
            );
        }
        // Launch a separate kernel to handle leftover elements
        if (tail > 0) {
            int blocks_tail = (tail + threads - 1) / threads;
            tail_sigmoid_kernel_float<<<blocks_tail, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                n_vec * 4,
                tail
            );
        }
    } else {
        int blocks = (size + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_scalar", ([&] {
            sigmoid_kernel_scalar<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Nondivergent Vectorized Sigmoid forward (CUDA)");
}
