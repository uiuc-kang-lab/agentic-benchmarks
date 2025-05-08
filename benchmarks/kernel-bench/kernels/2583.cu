#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using a grid-stride loop to reduce overhead
// and avoid unnecessary synchronizations since each thread operates independently
template <typename scalar_t>
__global__ void relu_kernel_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements in a grid-stride loop
    for (; idx < size; idx += stride) {
        __shared__ scalar_t shared_data[256];
        int local_idx = threadIdx.x;
        if (idx < size) shared_data[local_idx] = input[idx];
        __syncthreads();
        if (idx < size) output[idx] = (shared_data[local_idx] > 0) ? shared_data[local_idx] : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_optimized", ([&] {
        relu_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ReLU forward (CUDA)");
}
