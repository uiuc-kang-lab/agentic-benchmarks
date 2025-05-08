#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Shared memory version of the kernel for small inputs
__global__ void softsign_kernel_shared(const float* x, float* out, int num_elements) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < num_elements) {
        shared_data[tid] = x[gid];
        __syncthreads();
        
        out[gid] = shared_data[tid] / (1.0f + fabsf(shared_data[tid]));
    }
}

// Stream-based kernel for large inputs
__global__ void softsign_kernel_stream(const float* x, float* out, int offset, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int global_idx = offset + idx;
        out[global_idx] = x[global_idx] / (1.0f + fabsf(x[global_idx]));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int total_elements = x.numel();
    const int THRESHOLD = 1 << 20; // 1M elements threshold
    
    if (total_elements <= THRESHOLD) {
        // Use shared memory approach for small inputs
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        int shared_mem_size = threads * sizeof(float);
        
        softsign_kernel_shared<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), total_elements
        );
    } else {
        // Use multi-stream approach for large inputs
        int chunk_size = 1 << 20;
        int num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        int threads = 1024;
        
        const int MAX_STREAMS = 4; // Limit maximum number of streams
        int num_streams = std::min(num_chunks, MAX_STREAMS);
        std::vector<cudaStream_t> streams(num_streams);
        
        // Create streams
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // Launch kernels across streams in a round-robin fashion
        for (int i = 0; i < num_chunks; i++) {
            int stream_idx = i % num_streams;
            int offset = i * chunk_size;
            int current_chunk = std::min(chunk_size, total_elements - offset);
            int blocks = (current_chunk + threads - 1) / threads;
            
            softsign_kernel_stream<<<blocks, threads, 0, streams[stream_idx]>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), offset, current_chunk
            );
        }
        
        // Cleanup streams
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Softsign activation (CUDA)");
}