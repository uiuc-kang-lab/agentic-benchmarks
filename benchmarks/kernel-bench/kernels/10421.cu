#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_scalar(float xi) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x2 = xi * xi;
    float x3 = x2 * xi;
    float inner = sqrt_2_over_pi * (xi + coeff * x3);
    return 0.5f * xi * (1.f + tanhf(inner));
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    int totalVectors = n / 4;
    int vecChunk = (totalVectors + totalThreads - 1) / totalThreads;
    int startVec = tid * vecChunk;
    int endVec = min(startVec + vecChunk, totalVectors);

    for (int i = startVec; i < endVec; i++) {
        float4 in = reinterpret_cast<const float4*>(x)[i];
        float4 out;
        out.x = gelu_scalar(in.x);
        out.y = gelu_scalar(in.y);
        out.z = gelu_scalar(in.z);
        out.w = gelu_scalar(in.w);
        reinterpret_cast<float4*>(y)[i] = out;
    }

    int remainder_start = totalVectors * 4;
    int totalRemainder = n - remainder_start;
    int remChunk = (totalRemainder + totalThreads - 1) / totalThreads;
    int startRem = tid * remChunk;
    int endRem = min(startRem + remChunk, totalRemainder);

    for (int i = startRem; i < endRem; i++) {
        int idx = remainder_start + i;
        y[idx] = gelu_scalar(x[idx]);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = numSMs * 32;

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (n + num_streams - 1) / num_streams;
    chunk_size = (chunk_size + 3) & ~3;

    for (int i = 0; i < num_streams; i++) {
        int start = i * chunk_size;
        if (start >= n) break;
        int end = min(start + chunk_size, n);
        int elements = end - start;

        int blocks = (elements + threads - 1) / threads;
        blocks = min(blocks, max_blocks);

        gelu_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>() + start,
            y.data_ptr<float>() + start,
            elements
        );
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}