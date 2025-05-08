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

// Kernel using shared memory for optimized performance
__global__ void gelu_kernel(const float* x, float* y, int n) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int totalThreads = gridDim.x * blockDim.x;

    // Vectorized processing
    int totalVectors = n / 4;  // number of complete groups of 4
    int vectorStep = ceilf(float(totalVectors) / totalThreads);

    for (int i = global_tid; i < totalVectors; i += totalThreads) {
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        float4 val = x_vec[i];
        float4 res;
        res.x = gelu_scalar(val.x);
        res.y = gelu_scalar(val.y);
        res.z = gelu_scalar(val.z);
        res.w = gelu_scalar(val.w);
        shared_mem[tid] = reinterpret_cast<float*>(&res)[0];
        __syncthreads();

        if (i < n/4) {
            reinterpret_cast<float4*>(y)[i] = res;
        }
    }

    // Handle remaining elements that don't form a complete float4
    int remainder_start = totalVectors * 4;
    int totalRemainder = n - remainder_start;

    int remainderStep = ceilf(float(totalRemainder) / totalThreads);
    for (int i = global_tid; i < totalRemainder; i += totalThreads) {
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

    const int threads = 512; // Increase threads for more parallel processing
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = numSMs * 32;
    int blocks = (n + threads - 1) / threads;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    const int shared_mem_size = threads * sizeof(float); // Shared memory for storing intermediate results
    gelu_kernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}
