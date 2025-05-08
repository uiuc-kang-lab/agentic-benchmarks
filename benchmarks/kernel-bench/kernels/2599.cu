#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using warp-level primitives for optimization
// This kernel uses warp shuffle operations to optimize small reductions
// or specialized tasks within a warp

// Device function to apply ReLU
__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

// CUDA kernel using warp-level primitives
__global__ void relu_kernel_warp(float* __restrict__ output,
                                 const float* __restrict__ input,
                                 const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process elements using warp-level primitives
    for (; idx < size; idx += stride) {
        float val = input[idx];
        val = relu(val);

        // Use warp shuffle to broadcast the result within the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Write the result back to global memory
        if (threadIdx.x % 32 == 0) {
            output[idx] = relu(val);
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    relu_kernel_warp<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}