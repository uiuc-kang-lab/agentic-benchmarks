#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NUM_STREAMS 4

__global__ void softsign_kernel(const float4* __restrict__ x4,
                               float4* __restrict__ out4,
                               const int num_vectors,
                               const int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    while (idx < num_vectors) {
        float4 vec = x4[idx + offset];
        float4 res;
        
        res.x = vec.x / (1.0f + fabsf(vec.x));
        res.y = vec.y / (1.0f + fabsf(vec.y));
        res.z = vec.z / (1.0f + fabsf(vec.z));
        res.w = vec.w / (1.0f + fabsf(vec.w));
        
        out4[idx + offset] = res;
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int vectors_per_stream = (num_elements / 4) / NUM_STREAMS;
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256; // 256 is a multiple of 32, aligning with warp size
    const int blocks = 128; // Adjusted for H100
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int offset = i * vectors_per_stream;
        softsign_kernel<<<blocks, threads, 0, streams[i]>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            vectors_per_stream,
            offset
        );
    }
    
    // Handle remainder elements in the last stream
    const int remainder_start = NUM_STREAMS * vectors_per_stream * 4;
    const int remaining_vectors = (num_elements - remainder_start) / 4;
    if (remaining_vectors > 0) {
        softsign_kernel<<<blocks, threads, 0, streams[NUM_STREAMS-1]>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            remaining_vectors,
            NUM_STREAMS * vectors_per_stream
        );
    }
    
    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with stream overlap (CUDA)");
}