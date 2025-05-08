#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation optimized with __ldg() and 128-bit aligned vectorized accesses.
// For float types, we use float4 (128 bits = 4 * 32-bit), and for double types, we use double2 (128 bits = 2 * 64-bit).

template <typename scalar_t>
__global__ void relu_kernel_ldg_aligned(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Specialized implementation for float using float4 vectorized operations
    if constexpr (sizeof(scalar_t) == 4) {
        int total_vec = size / 4;  // Number of 4-element vectors
        float4* out_vec = reinterpret_cast<float4*>(output);
        const float4* in_vec = reinterpret_cast<const float4*>(input);

        // Process vectorized portion
        for (int i = idx; i < total_vec; i += stride) {
            // Use __ldg to load read-only data from global memory
            float4 in_val = __ldg(&in_vec[i]);
            float4 out_val;
            out_val.x = in_val.x > 0.0f ? in_val.x : 0.0f;
            out_val.y = in_val.y > 0.0f ? in_val.y : 0.0f;
            out_val.z = in_val.z > 0.0f ? in_val.z : 0.0f;
            out_val.w = in_val.w > 0.0f ? in_val.w : 0.0f;
            out_vec[i] = out_val;
        }

        // Process any remaining elements that don't form a full float4
        int rem_start = total_vec * 4;
        for (int i = rem_start + idx; i < size; i += stride) {
            float tmp = __ldg(&input[i]);
            output[i] = tmp > 0.0f ? tmp : 0.0f;
        }
    } else {
        // Implementation for double types using double2 vectorized operations (128-bit = 2 x 64-bit)
        int total_vec = size / 2;  // Number of 2-element vectors
        double2* out_vec = reinterpret_cast<double2*>(output);
        const double2* in_vec = reinterpret_cast<const double2*>(input);

        // Process vectorized portion
        for (int i = idx; i < total_vec; i += stride) {
            double2 in_val = __ldg(&in_vec[i]);
            double2 out_val;
            out_val.x = in_val.x > 0.0 ? in_val.x : 0.0;
            out_val.y = in_val.y > 0.0 ? in_val.y : 0.0;
            out_vec[i] = out_val;
        }

        // Process remaining element if size is odd
        int rem_start = total_vec * 2;
        for (int i = rem_start + idx; i < size; i += stride) {
            double tmp = __ldg(&input[i]);
            output[i] = tmp > 0.0 ? tmp : 0.0;
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    // Launch configuration based on the total number of elements (scalar count)
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_ldg_aligned", ([&] {
        relu_kernel_ldg_aligned<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with __ldg() and 128-bit aligned accesses (CUDA)");
}
