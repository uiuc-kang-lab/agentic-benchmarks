#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NUM_STREAMS 4

__global__ void softsign_kernel_streamed(const float* __restrict__ x, 
                                       float* __restrict__ out, 
                                       int chunk_size,
                                       int offset) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) {
        const int global_idx = tid + offset;
        float val = x[global_idx];
        float abs_val = fabsf(val);
        // Use faster division intrinsic since softsign doesn't require full precision
        out[global_idx] = __fdividef(val, (1.0f + abs_val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Process chunks in parallel using different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int offset = i * chunk_size;
        const int current_chunk_size = min(chunk_size, num_elements - offset);
        if (current_chunk_size <= 0) break;
        
        const int blocks = (current_chunk_size + threads - 1) / threads;
        
        softsign_kernel_streamed<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            current_chunk_size,
            offset
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with stream pipelining (CUDA)");
}