#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using vectorized loads
template <typename scalar_t>
__global__ void relu_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4* in_vec = (float4*)(&input[idx]);
        float4* out_vec = (float4*)(&output[idx]);
        float4 val = *in_vec;
        
        val.x = val.x > 0 ? val.x : 0;
        val.y = val.y > 0 ? val.y : 0;
        val.z = val.z > 0 ? val.z : 0;
        val.w = val.w > 0 ? val.w : 0;
        
        *out_vec = val;
    }
    // Handle remaining elements
    else if (idx < size) {
        for (int i = 0; i < 4 && idx + i < size; i++) {
            scalar_t val = input[idx + i];
            output[idx + i] = val > 0 ? val : 0;
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_vectorized", ([&] {
        relu_kernel_vectorized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward vectorized (CUDA)");
}