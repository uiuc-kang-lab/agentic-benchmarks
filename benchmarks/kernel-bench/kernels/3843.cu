#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function to compute softplus in a numerically stable way
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    if (x > 20.0) {
        return x;
    } else if (x < -20.0) {
        return exp(x);
    } else {
        return log1p(exp(x));
    }
}

// Kernel using shared memory for reduction
template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    extern __shared__ scalar_t shared_data[];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    scalar_t sum = 0;
    for (int i = idx; i < size; i += stride) {
        sum += compute_softplus(input[i]);
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}