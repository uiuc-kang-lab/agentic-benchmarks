#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Configuration constants
const int THREADS_PER_BLOCK = 256;
const int VECTOR_SIZE = 4;  // For float4 vectorization

// Optimized kernel to minimize warp divergence
template <typename scalar_t>
__global__ void relu_kernel_warp_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Vector-width aligned processing
    if constexpr (std::is_same_v<scalar_t, float>) {
        const int vector_idx = idx * VECTOR_SIZE;
        float4* in4 = (float4*)input;
        float4* out4 = (float4*)output;

        for (int i = vector_idx; i < size / VECTOR_SIZE; i += stride) {
            float4 val = in4[i];
            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);
            out4[i] = val;
        }

        // Handle remaining elements
        for (int i = vector_idx + size / VECTOR_SIZE * VECTOR_SIZE; i < size; i += stride) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    } else {
        for (int i = idx; i < size; i += stride) {
            output[i] = max(input[i], static_cast<scalar_t>(0));
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t total_size = input.numel();
    const int blocks = (total_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_warp_optimized", ([&] {
        relu_kernel_warp_optimized<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Divergence Optimized ReLU forward (CUDA)");
}