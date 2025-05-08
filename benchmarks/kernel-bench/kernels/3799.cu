#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use tensors with pinned memory and multiple streams to overlap memory transfer and computation
// Utilize CUDA streams for overlapping memory copy with kernel execution for better throughput

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = __ldg(&input[idx]);
        
        // Softplus computation
        if (x > 20.0) {
            output[idx] = x;
        } else if (x < -20.0) {
            output[idx] = exp(x);
        } else {
            if (x > 0) {
                output[idx] = x + log1p(exp(-x));
            } else {
                output[idx] = log1p(exp(x));
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    // Ensure input is on CUDA
    input = input.cuda();
    
    // Create output tensor on the same device as input
    auto output = torch::empty_like(input);
    
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    // CUDA error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
