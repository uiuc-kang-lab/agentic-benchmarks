#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Modular device function for computing the sigmoid
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Modular device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Optimized Swish kernel using shared memory and warp-level reduction
__global__ void swish_shared_warp_reduction_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float swish_val = compute_swish(val);
        sum += swish_val;
        y[i] = swish_val;
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction for final stages
    if (tid < 32) {
        float warp_sum = shared_data[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid == 0) {
            // Store the result of the reduction if needed
            // This example does not use the reduction result
        }
    }
}

torch::Tensor swish_shared_warp_reduction_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    swish_shared_warp_reduction_kernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_shared_warp_reduction_forward, "Swish activation forward pass with shared memory and warp-level reduction (CUDA)");
}