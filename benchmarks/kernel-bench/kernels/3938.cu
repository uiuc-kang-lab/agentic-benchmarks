#include <torch/extension.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Utility macros for error checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Number of CUDA streams to partition the work
#define NUM_STREAMS 4

// This kernel processes a chunk of the data using a grid-stride loop 
// for improved coalesced memory access and load balancing.
__global__ void softsign_kernel_streamed_coalesced(const float* __restrict__ x, 
                                                   float* __restrict__ out, 
                                                   int chunk_size, 
                                                   int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Use grid-stride loop so that threads can cover the entire chunk
    for (int i = idx; i < chunk_size; i += stride) {
        int global_idx = offset + i;
        float val = x[global_idx];
        out[global_idx] = val / (1.0f + fabsf(val));
    }
}

// Host function launches kernels on different CUDA streams. The input tensor
// is partitioned into NUM_STREAMS chunks. Each kernel invocation processes
// its chunk using a grid-stride loop, ensuring coalesced memory accesses.
// This approach combines stream pipelining with load-balanced kernel design.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    const int num_elements = x.numel();

    // Determine chunk sizes for each stream
    // This method divides the work evenly and handles any remainder
    int base_chunk = num_elements / NUM_STREAMS;
    int remainder = num_elements % NUM_STREAMS;

    const int threads = 256;
    
    // Create multiple CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int offset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Distribute the remainder among the first few streams
        int current_chunk = base_chunk + (i < remainder ? 1 : 0);
        if (current_chunk <= 0) break;
        int blocks = (current_chunk + threads - 1) / threads;
        
        // Launch the kernel on the current stream with a grid-stride loop
        softsign_kernel_streamed_coalesced<<<blocks, threads, 0, streams[i]>>> (
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            current_chunk,
            offset
        );
        
        offset += current_chunk;
    }
    
    // Synchronize and clean up streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with streamed, coalesced grid-stride kernel (CUDA)");
}
