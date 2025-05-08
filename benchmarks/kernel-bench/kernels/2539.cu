#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation with loop unrolling
template <typename scalar_t>
__global__ void relu_kernel_unroll(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes 4 elements per iteration using manual loop unrolling
    for (int i = idx; i < size; i += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int index = i + j * stride;
            if (index < size) {
                scalar_t in_val = input[index];
                output[index] = in_val > static_cast<scalar_t>(0) ? in_val : static_cast<scalar_t>(0);
            }
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_unroll", ([&] {
        relu_kernel_unroll<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with loop unrolling (CUDA)");
}
