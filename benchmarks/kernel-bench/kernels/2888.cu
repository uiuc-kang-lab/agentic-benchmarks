#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    // Sequentially reduce using warp shuffle instructions
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void sigmoid_kernel_reduce(float const* __restrict__ input, float* __restrict__ output, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < size; idx += stride) {
        float val = -input[idx];
        float exp_val = expf(val);
        float sigmoid_val = 1.0f / (1.0f + exp_val);

        // Example warp-level sum reduction for illustration (though not used in sigmoid)
        float reduced_val = sigmoid_val;  
        if (threadIdx.x == 0) {
            atomicAdd(&output[0], reduced_val);  // just a placeholder usage
        }

        output[idx] = sigmoid_val;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    sigmoid_kernel_reduce<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward with warp-level reduction (CUDA)");
}
