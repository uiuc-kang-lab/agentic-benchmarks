#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with adjusted indexing
__global__ void tanh_index_optimized_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_index_optimized_kernel", ([&] {
        tanh_index_optimized_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            numel
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Index Optimized Tanh forward (CUDA)");
}