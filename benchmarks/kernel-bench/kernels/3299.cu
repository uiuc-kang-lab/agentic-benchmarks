#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that uses shared memory for warp-level reductions
__global__ void swish_shared_reduction_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float temp = 0.0f;

    for (int i = globalIdx; i < n; i += stride) {
        float val = x[i];
        temp += val * (1.0f / (1.0f + expf(-val)));
    }

    sdata[tid] = temp;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Output result for this block's sum into global memory
    if (tid == 0) y[blockIdx.x] = sdata[0];
}

// Finalize the global reduction and normalize across blocks
__global__ void finalize_reduction(float* y, int numBlocks) {
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        sum += y[i];
    }
    y[0] = sum;
}

torch::Tensor swish_reduction_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty({1}, x.options());
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Allocate temporary buffer for inter-block results
    auto tempY = torch::empty({blocks}, x.options());

    swish_shared_reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        tempY.data_ptr<float>(),
        n
    );

    finalize_reduction<<<1, 1>>>(tempY.data_ptr<float>(), blocks);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_reduction_forward, "Swish activation forward with shared memory reduction (CUDA)");
}
