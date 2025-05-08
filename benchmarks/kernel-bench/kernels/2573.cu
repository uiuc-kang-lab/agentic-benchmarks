#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level ReLU kernel using warp shuffle for small reductions
// This kernel is optimized for warp-level operations, reducing shared memory usage

// Warp size is typically 32 for most NVIDIA GPUs
constexpr int WARP_SIZE = 32;

// Warp-level ReLU operation
__device__ float warp_relu(float val) {
    return val > 0 ? val : 0;
}

// Kernel using warp-level primitives
template <typename scalar_t>
__global__ void warp_optimized_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements in a grid-stride loop
    for (int i = idx; i < size; i += stride) {
        scalar_t val = input[i];
        val = warp_relu(val);

        // Use warp shuffle to perform a reduction or specialized task
        // Here, we simply demonstrate a warp shuffle operation
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Store the result back
        output[i] = val;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_optimized_relu_kernel", ([&] {
        warp_optimized_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Optimized ReLU forward (CUDA)");
}
