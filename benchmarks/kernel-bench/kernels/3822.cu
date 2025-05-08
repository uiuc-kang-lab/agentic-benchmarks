#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Compute softplus using a stable and branchless formula
__device__ __forceinline__ float compute_softplus(float x) {
    float ax = fabsf(x);
    float max_val = (x + ax) * 0.5f;
    return max_val + log1pf(expf(-ax));
}

// Vectorized load and compute for float using a modular approach
__device__ __forceinline__ void vectorized_softplus(const float4* input, float4* output) {
    float4 in_val = __ldg(input);
    output->x = compute_softplus(in_val.x);
    output->y = compute_softplus(in_val.y);
    output->z = compute_softplus(in_val.z);
    output->w = compute_softplus(in_val.w);
}

// Unified kernel to compute softplus on inputs, improving modularity and clarity
// Supports both vectors and scalars efficiently
template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if constexpr (std::is_same<scalar_t, float>::value) {
        uintptr_t input_addr = reinterpret_cast<uintptr_t>(input);
        bool is_aligned = (input_addr % (4 * sizeof(float))) == 0;

        if (is_aligned) {
            int vecSize = size / 4;
            for (int i = tid; i < vecSize; i += stride) {
                const float4* in_ptr = reinterpret_cast<const float4*>(input + i * 4);
                float4 out_val;
                vectorized_softplus(in_ptr, &out_val);
                *reinterpret_cast<float4*>(output + i * 4) = out_val;
            }

            int tail_index = vecSize * 4;
            for (int i = tail_index + tid; i < size; i += stride) {
                output[i] = compute_softplus(__ldg(&input[i]));
            }
        } else {
            for (int i = tid; i < size; i += stride) {
                output[i] = compute_softplus(__ldg(&input[i]));
            }
        }
    } else {
        for (int i = tid; i < size; i += stride) {
            scalar_t x = __ldg(&input[i]);
            scalar_t ax = fabs(x);
            scalar_t max_val = (x + ax) * static_cast<scalar_t>(0.5);
            output[i] = max_val + log1p(exp(-ax));
        }
    }
}

// Function definition for the module, calling the modular kernel
// that computes the Softplus function

// Compute the softplus output using preallocated tensors
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}