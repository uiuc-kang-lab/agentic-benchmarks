#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 3 < size) {
        float4 in = reinterpret_cast<const float4*>(input)[idx/4];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[idx/4] = out;
    }
    else {
        for (int i = 0; i < 4 && idx + i < size; ++i) {
            output[idx + i] = tanhf(input[idx + i]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (input.numel() + elements_per_block - 1) / elements_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_optimized_kernel", ([&] {
        tanh_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh forward (CUDA)");
}