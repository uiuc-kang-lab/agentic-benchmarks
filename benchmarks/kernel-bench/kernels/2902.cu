#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    extern __shared__ scalar_t shared_input[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        shared_input[threadIdx.x] = input[idx];
        __syncthreads();
        
        output[idx] = tanhf(shared_input[threadIdx.x]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    const int shared_memory_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel_shared", ([&] {
        tanh_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with shared memory (CUDA)");
}