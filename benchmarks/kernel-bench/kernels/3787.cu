#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    __shared__ scalar_t shared_data[256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    scalar_t val = 0.0;
    if (gid < size) {
        const scalar_t x = input[gid];
        if (x > 20.0) {
            val = x;
        } else if (x < -20.0) {
            val = exp(x);
        } else {
            val = log1p(exp(x));
        }
        shared_data[tid] = val;
    }
    __syncthreads();
    
    // Write result to global memory
    if (gid < size) {
        output[gid] = shared_data[tid];
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}