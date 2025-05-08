#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Compute GELU for a single scalar value
__device__ __forceinline__ float gelu_scalar(float xi) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x2 = xi * xi;
    float x3 = x2 * xi;
    float inner = sqrt_2_over_pi * (xi + coeff * x3);
    return 0.5f * xi * (1.f + tanhf(inner));
}

// Kernel that uses shared memory for vectorized processing
__global__ void gelu_kernel(const float* x, float* y, const int n) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Process data in chunks to handle vectorized loads
    for (int idx = global_tid; idx < n; idx += totalThreads) {
        // Check if we can do a vectorized load (if we have 4 elements remaining and aligned address)
        if (idx + 3 < n && (idx % 4) == 0) {
            float4 in = *reinterpret_cast<const float4*>(&x[idx]);
            float4 out;
            out.x = gelu_scalar(in.x);
            out.y = gelu_scalar(in.y);
            out.z = gelu_scalar(in.z);
            out.w = gelu_scalar(in.w);
            *reinterpret_cast<float4*>(&y[idx]) = out;
        } else if (idx < n) {
            // Handle non-vectorized elements
            y[idx] = gelu_scalar(x[idx]);
        }
    }

    // Handle remaining elements that don't form a complete float4
    int remainder_start = totalVectors * 4;
    int totalRemainder = n - remainder_start;
    int remChunk = (totalRemainder + totalThreads - 1) / totalThreads;
    int startRem = global_tid * remChunk;
    int endRem = startRem + remChunk;
    if (endRem > totalRemainder) endRem = totalRemainder;

    for (int i = startRem; i < endRem; i++) {
        int idx = remainder_start + i;
        if (idx < n) {
            y[idx] = gelu_scalar(x[idx]);
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = numSMs * 32;
    int blocks = (n + threads - 1) / threads;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    // Allocate shared memory for float elements
    int shared_mem_size = threads * sizeof(float);

    gelu_kernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}
