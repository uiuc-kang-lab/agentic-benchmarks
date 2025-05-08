#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_unrolled(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 4;
    
    #pragma unroll
    for (int idx = tid; idx < size; idx += stride) {
        scalar_t val1 = input[idx];
        scalar_t val2 = input[idx + stride];
        scalar_t val3 = input[idx + stride * 2];
        scalar_t val4 = input[idx + stride * 3];
        
        output[idx] = val1 > 0 ? val1 : 0;
        output[idx + stride] = val2 > 0 ? val2 : 0;
        output[idx + stride * 2] = val3 > 0 ? val3 : 0;
        output[idx + stride * 3] = val4 > 0 ? val4 : 0;
    }
    
    // Handle remaining elements
    for (int idx = tid + size - (size % (stride * unroll_factor)); idx < size; idx += stride) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_unrolled", ([&] {
        relu_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}