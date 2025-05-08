#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store frequently accessed constants in constant memory
__constant__ float k_sqrt_2_over_pi = 0.7978845608f;
__constant__ float k_coeff = 0.044715f;

// Compute GELU activation for a single float using constant memory constants
__device__ __forceinline__ float gelu_scalar(float x) {
    float x_cubed = x * x * x;
    float inner = k_sqrt_2_over_pi * (x + k_coeff * x_cubed);
    return 0.5f * x * (1.f + tanhf(inner));
}

// CUDA kernel using vectorized loads/stores with constant memory for constants
__global__ void gelu_kernel(const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Process vectorized portion with float4 loads/stores
    int n_vectors = n / 4;  // number of complete groups of 4 floats
    for (int i = tid; i < n_vectors; i += totalThreads) {
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        float4* y_vec = reinterpret_cast<float4*>(y);
        float4 in = x_vec[i];
        float4 out;
        out.x = gelu_scalar(in.x);
        out.y = gelu_scalar(in.y);
        out.z = gelu_scalar(in.z);
        out.w = gelu_scalar(in.w);
        y_vec[i] = out;
    }

    // Process remaining elements that don't form a complete float4
    int rem_start = n_vectors * 4;
    if (tid < (n - rem_start)) {
        int idx = rem_start + tid;
        y[idx] = gelu_scalar(x[idx]);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = num_sm * 32;  // heuristic for max blocks
    int blocks = (n + threads - 1) / threads;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}
