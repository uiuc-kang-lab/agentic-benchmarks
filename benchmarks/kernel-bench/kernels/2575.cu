#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements ReLU in a vectorized grid-stride loop without using atomic operations,
// since each thread processes distinct elements. Atomic operations are omitted to avoid
// unnecessary global memory contention and overhead.

template <typename scalar_t>
__global__ void atomic_minimal_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // For 32-bit floats, use vectorized operations with float4 if possible.
    if constexpr (sizeof(scalar_t) == 4) {
        int vec_elements = size / 4; // each float4 holds 4 floats
        float4* in_vec = reinterpret_cast<float4*>(const_cast<scalar_t*>(input));
        float4* out_vec = reinterpret_cast<float4*>(output);
        
        // Process vectorized portion
        for (int i = idx; i < vec_elements; i += stride) {
            float4 val = __ldg(&in_vec[i]);
            val.x = (val.x > 0.0f) ? val.x : 0.0f;
            val.y = (val.y > 0.0f) ? val.y : 0.0f;
            val.z = (val.z > 0.0f) ? val.z : 0.0f;
            val.w = (val.w > 0.0f) ? val.w : 0.0f;
            out_vec[i] = val;
        }

        // Process remaining elements that don't fit into a float4
        int base = vec_elements * 4;
        for (int i = base + idx; i < size; i += stride) {
            float tmp = __ldg(&input[i]);
            output[i] = (tmp > 0.0f) ? tmp : 0.0f;
        }
    } else if constexpr (sizeof(scalar_t) == 8) {
        // For 64-bit types (double), use vectorized operations with double2 if possible.
        int vec_elements = size / 2; // each double2 holds 2 doubles
        double2* in_vec = reinterpret_cast<double2*>(const_cast<scalar_t*>(input));
        double2* out_vec = reinterpret_cast<double2*>(output);

        // Process vectorized portion
        for (int i = idx; i < vec_elements; i += stride) {
            double2 val = __ldg(&in_vec[i]);
            val.x = (val.x > 0.0) ? val.x : 0.0;
            val.y = (val.y > 0.0) ? val.y : 0.0;
            out_vec[i] = val;
        }

        // Process remaining element if the size is odd
        int base = vec_elements * 2;
        for (int i = base + idx; i < size; i += stride) {
            double tmp = __ldg(&input[i]);
            output[i] = (tmp > 0.0) ? tmp : 0.0;
        }
    } else {
        // Fallback for other types
        for (int i = idx; i < size; i += stride) {
            scalar_t val = input[i];
            output[i] = (val > static_cast<scalar_t>(0)) ? val : static_cast<scalar_t>(0);
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_minimal_relu_kernel", ([&] {
        atomic_minimal_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Elementwise ReLU forward (CUDA) without unnecessary atomics");
}
