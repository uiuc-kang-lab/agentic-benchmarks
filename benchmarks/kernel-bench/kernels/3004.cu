#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with atomic operations for race condition handling
__global__ void tanh_atomic_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Shared memory for reducing global memory atomic operations
    __shared__ float shared_output[512];

    for (int i = idx; i < size; i += stride) {
        float val = tanhf(input[i]);
        atomicAdd(&shared_output[threadIdx.x], val);
    }
    __syncthreads();

    // Atomic add to global memory from shared memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            atomicAdd(&output[blockIdx.x], shared_output[i]);
        }
    }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int numel = input.numel();
    const int threads = 512;
    const int blocks = (numel + threads - 1) / threads;

    tanh_atomic_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh with atomic operations (CUDA)");
}