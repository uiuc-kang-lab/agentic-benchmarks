#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void elu_kernel_stream(const float4* x, float4* out, float alpha, int n4, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = x[idx + offset];
        float4 result;
        result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
        result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
        result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
        result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
        out[idx + offset] = result;
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;  // number of float4 elements
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    int chunk_size = (n4 + NUM_STREAMS - 1) / NUM_STREAMS;
    chunk_size = (chunk_size + threads - 1) / threads * threads; // Round up to multiple of thread block size
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        if (offset < n4) {
            int current_chunk_size = min(chunk_size, n4 - offset);
            int blocks = (current_chunk_size + threads - 1) / threads;
            
            elu_kernel_stream<<<blocks, threads, 0, streams[i]>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                alpha,
                current_chunk_size,
                offset
            );
        }
    }
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "Pipelined ELU activation (CUDA)");
}