#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Branchless computation of softplus using the stable formulation:
// softplus(x) = max(x, 0) + log1p(exp(-|x|))
// Here max(x, 0) is computed as (x + |x|) * 0.5 to avoid conditional branches.
__device__ __forceinline__ float softplus_branchless(float x) {
    float ax = fabsf(x);
    float max_val = (x + ax) * 0.5f;
    return max_val + log1pf(expf(-ax));
}

// CUDA kernel for softplus that minimizes warp divergence by using a uniform branchless formula.
// For float data, we utilize vectorized memory accesses using float4 for better throughput.
// For other types, we fall back to a scalar loop with the same branchless computation.

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if constexpr (std::is_same<scalar_t, float>::value) {
        // Check if input pointer is aligned for float4
        uintptr_t input_addr = reinterpret_cast<uintptr_t>(input);
        bool is_aligned = (input_addr % (4 * sizeof(float))) == 0;
        
        if (is_aligned) {
            // Process elements in chunks of 4 using vectorized loads/stores
            int vecSize = size / 4;
            for (int i = tid; i < vecSize; i += stride) {
                // Use vectorized load
                const float4* in_ptr = reinterpret_cast<const float4*>(input + i * 4);
                float4 in_val = __ldg(in_ptr);
                float4 out_val;
                out_val.x = softplus_branchless(in_val.x);
                out_val.y = softplus_branchless(in_val.y);
                out_val.z = softplus_branchless(in_val.z);
                out_val.w = softplus_branchless(in_val.w);
                // Use vectorized store
                *reinterpret_cast<float4*>(output + i * 4) = out_val;
            }

            // Handle remaining elements
            int tail_index = vecSize * 4;
            for (int i = tail_index + tid; i < size; i += stride) {
                float x = __ldg(&input[i]);
                output[i] = softplus_branchless(x);
            }
        } else {
            // Fall back to scalar processing if not aligned
            for (int i = tid; i < size; i += stride) {
                float x = __ldg(&input[i]);
                output[i] = softplus_branchless(x);
            }
        }
    } else {
        // For non-float types, use scalar processing with branchless computation
        for (int i = tid; i < size; i += stride) {
            scalar_t x = __ldg(&input[i]);
            scalar_t ax = fabs(x);
            scalar_t max_val = (x + ax) * static_cast<scalar_t>(0.5);
            output[i] = max_val + log1p(exp(-ax));
        }
    }
}

// CUDA forward function
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
