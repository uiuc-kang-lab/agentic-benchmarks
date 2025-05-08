#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Grid-stride loop pattern
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements in a strided fashion
    for (int idx = tid; idx < size; idx += stride) {
        const scalar_t x = input[idx];
        output[idx] = compute_softplus(x);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Use smaller thread blocks for better occupancy
    const int threads = 128;
    
    // Calculate optimal number of blocks based on SM count
    int multiProcessorCount;
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, 0);
    // Aim for 4 blocks per SM for better distribution
    const int blocks = min(4 * multiProcessorCount, (size + threads - 1) / threads);

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