#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute softplus in a numerically stable way
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    if (x > 20.0) {
        return x;
    } else if (x < -20.0) {
        return exp(x);
    } else {
        const scalar_t exp_x = exp(x);
        return log1p(exp_x);
    }
}

// Kernel using shared memory to reduce global memory accesses
template <typename scalar_t>
__global__ void softplus_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    extern __shared__ scalar_t shared_data[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Compute softplus using shared memory
    if (idx < size) {
        scalar_t x = shared_data[tid];
        output[idx] = compute_softplus(x);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}