#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>

// Branchless softplus using the formulation: softplus(x) = max(x, 0) + log1p(exp(-|x|))
// This formulation avoids explicit divergent branches.

// Overload for float
__device__ __forceinline__ float softplus_val(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

// Overload for double
__device__ __forceinline__ double softplus_val(double x) {
    return fmax(x, 0.0) + log1p(exp(-fabs(x)));
}

// CUDA kernel with branchless softplus computation
// For float types, we use vectorized loads/stores via float4 to improve memory throughput.

template <typename scalar_t>
__global__ void softplus_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if constexpr (std::is_same<scalar_t, float>::value) {
        int vec_size = size / 4;  // number of float4 groups
        int vec_idx = idx;
        while (vec_idx < vec_size) {
            // Load 4 floats at once
            float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + vec_idx);
            float4 out_vec;
            out_vec.x = softplus_val(in_vec.x);
            out_vec.y = softplus_val(in_vec.y);
            out_vec.z = softplus_val(in_vec.z);
            out_vec.w = softplus_val(in_vec.w);
            reinterpret_cast<float4*>(output)[vec_idx] = out_vec;
            vec_idx += stride;
        }
        int offset = vec_size * 4;
        for (int i = idx; i < size - offset; i += stride) {
            output[offset + i] = softplus_val(input[offset + i]);
        }
    } else {
        // For double or other floating types, process element-wise
        for (int i = idx; i < size; i += stride) {
            output[i] = softplus_val(input[i]);
        }
    }
}

// Torch forward function
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
