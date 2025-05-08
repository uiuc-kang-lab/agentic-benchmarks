#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_stream(const float* __restrict__ in, 
                                     float* __restrict__ out,
                                     const int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        out[tid] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int chunk_size = 128 * 1024; // Optimized chunk size
    const int num_chunks = (num_elements + chunk_size - 1) / chunk_size;
    
    // Create streams and events for pipelining
    const int num_streams = 3; // Triple buffering
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<cudaEvent_t> events(num_streams);
    
    // Initialize streams and events
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }
    
    const int threads = 256;
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    // Pipeline execution across chunks
    for (int i = 0; i < num_chunks; i++) {
        const int stream_idx = i % num_streams;
        const int offset = i * chunk_size;
        const int current_size = std::min(chunk_size, num_elements - offset);
        const int blocks = (current_size + threads - 1) / threads;
        
        // Launch kernel in stream
        softsign_kernel_stream<<<blocks, threads, 0, streams[stream_idx]>>>(
            in_ptr + offset,
            out_ptr + offset,
            current_size
        );
        
        // Record event for synchronization
        cudaEventRecord(events[stream_idx], streams[stream_idx]);
        
        // If we're about to reuse a stream, ensure its previous work is done
        if (i >= num_streams) {
            const int wait_idx = (i - num_streams) % num_streams;
            cudaStreamWaitEvent(streams[stream_idx], events[wait_idx]);
        }
    }
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined Softsign activation (CUDA)");
}