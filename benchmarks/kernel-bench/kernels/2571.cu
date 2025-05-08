#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with memory coalescing optimization
// Ensures that threads in a warp read/write consecutive memory locations
// Uses float4 for vectorized access when possible

template <typename scalar_t>
__global__ void coalesced_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Vectorized processing for aligned data
    if constexpr (std::is_same_v<scalar_t, float>) {
        const int vector_elements = size / 4;
        float4* in4 = (float4*)input;
        float4* out4 = (float4*)output;

        for (int i = tid; i < vector_elements; i += stride) {
            float4 val = in4[i];
            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);
            out4[i] = val;
        }

        // Process remaining elements
        const int scalar_start = vector_elements * 4 + tid;
        for (int i = scalar_start; i < size; i += stride) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    } else {
        for (int i = tid; i < size; i += stride) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_relu_kernel", ([&] {
        coalesced_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Access ReLU forward (CUDA)");
}