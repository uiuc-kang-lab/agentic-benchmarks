#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Threshold to decide whether to use multi-stream processing
const int CHUNK_THRESHOLD = 1 << 20; // 1M elements

// Unified kernel that can optionally use shared memory
// When 'use_shared' is true, the kernel loads data into shared memory before computing softsign.
// Otherwise, it computes directly from global memory.
__global__ void softsign_kernel_adaptive(const float* __restrict__ x, 
                                           float* __restrict__ out, 
                                           int offset, 
                                           int num_elements, 
                                           bool use_shared) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int global_idx = offset + idx;
        float val;
        if (use_shared) {
            extern __shared__ float shared_data[];
            // Load the input value into shared memory
            shared_data[threadIdx.x] = x[global_idx];
            __syncthreads();
            val = shared_data[threadIdx.x];
        } else {
            // Directly load from global memory
            val = x[global_idx];
        }
        out[global_idx] = val / (1.0f + fabsf(val));
    }
}

// Host function exposed to Python
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int total_elements = x.numel();

    // For small inputs, use shared memory for better memory coalescing
    if (total_elements <= CHUNK_THRESHOLD) {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        int shared_mem_size = threads * sizeof(float);
        bool use_shared = true;
        softsign_kernel_adaptive<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), 0, total_elements, use_shared);
    } else {
        // For large inputs, use a multi-stream approach to overlap kernel execution
        int chunk_size = CHUNK_THRESHOLD;
        int num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        int threads = 1024;
        const int MAX_STREAMS = 4;
        int num_streams = std::min(num_chunks, MAX_STREAMS);

        // Create CUDA streams
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        bool use_shared = false; // Avoid extra shared memory overhead on large inputs
        for (int i = 0; i < num_chunks; i++) {
            int stream_idx = i % num_streams;
            int offset = i * chunk_size;
            int current_chunk = std::min(chunk_size, total_elements - offset);
            int blocks = (current_chunk + threads - 1) / threads;
            // Launch the kernel chunk asynchronously on the corresponding stream
            softsign_kernel_adaptive<<<blocks, threads, 0, streams[stream_idx]>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), offset, current_chunk, use_shared);
        }

        // Synchronize and destroy streams
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive and Unified Softsign activation (CUDA) combining shared memory and stream optimizations");
}
