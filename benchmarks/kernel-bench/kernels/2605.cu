#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined CUDA kernel utilizing both grid-stride loop and vectorization
// to optimize for both memory overhead and computation speed
template <typename scalar_t>
__global__ void relu_kernel_combined(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    
    using vec_t = float4;
    if (size % vec_size == 0) { // If size is aligned with vector size
        vec_t* in_vec = (vec_t*)input;
        vec_t* out_vec = (vec_t*)output;
        const int vec_elements = size / vec_size;

        for (int i = tid; i < vec_elements; i += stride) {
            vec_t val = in_vec[i];
            val.x = fmaxf(val.x, 0);
            val.y = val.y > 0 ? val.y : 0;
            val.z = val.z > 0 ? val.z : 0;
            val.w = val.w > 0 ? val.w : 0;
            out_vec[i] = val;
        }
    } else { // Use grid-stride for remaining elements or if not aligned with vec_size
        for (int idx = tid; idx < size; idx += stride) {
            scalar_t x = input[idx];
            output[idx] = (x > 0) ? x : static_cast<scalar_t>(0);
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_combined", ([&] {
        relu_kernel_combined<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined ReLU forward (CUDA)");
}
