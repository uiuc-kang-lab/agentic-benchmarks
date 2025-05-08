#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel_optimized(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use grid-stride loop to handle arrays larger than grid size
    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        // Cache the computation of expf(-val) to reduce floating-point overhead
        float exp_val = expf(-val);
        output[i] = 1.0f / (1.0f + exp_val);
    }
}



torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    sigmoid_kernel_syncthreads<<<blocks, threads, shared_mem_size>>>( 
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) with minimal __syncthreads usage");
}
