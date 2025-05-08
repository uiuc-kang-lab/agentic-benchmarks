#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Optimized CUDA kernel using warp-level primitives for reduction
__global__ void gelu_kernel_warp_optimized(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < numel; i += stride) {
        float val = input[i];
        float gelu_val = gelu_function(val);
        
        // Warp-level reduction (example, not needed for this operation)
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            gelu_val += __shfl_down_sync(0xFFFFFFFF, gelu_val, offset);
        }

        // Assign result
        if (threadIdx.x % warpSize == 0) {
            output[i] = gelu_val;  // Storing only the result of the warp leader
        }
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for the warp-optimized version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    // Launch the optimized kernel
    gelu_kernel_warp_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with warp-level optimization");
}