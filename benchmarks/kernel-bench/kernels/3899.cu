#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized shared memory kernel with vectorized loads/stores
__global__ void softsign_kernel_shared_vec4(const float4* x4, float4* out4, int num_elements4) {
    extern __shared__ float4 shared_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < num_elements4) {
        // Vectorized load
        shared_data[tid] = x4[gid];
        __syncthreads();
        
        float4 val = shared_data[tid];
        float4 result;
        
        // Process all 4 elements
        result.x = val.x / (1.0f + fabsf(val.x));
        result.y = val.y / (1.0f + fabsf(val.y));
        result.z = val.z / (1.0f + fabsf(val.z));
        result.w = val.w / (1.0f + fabsf(val.w));
        
        // Vectorized store
        out4[gid] = result;
    }
}

// Stream-based kernel for remaining elements
__global__ void softsign_kernel_remainder(const float* x, float* out, int offset, int num_elements) {
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
        // Process elements in chunks of 4 using vectorized operations
        int num_elements4 = total_elements / 4;
        int remainder = total_elements % 4;
        
        if (num_elements4 > 0) {
            int threads = 256;
            int blocks = (num_elements4 + threads - 1) / threads;
            int shared_mem_size = threads * sizeof(float4);
            
            softsign_kernel_shared_vec4<<<blocks, threads, shared_mem_size>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                reinterpret_cast<float4*>(out.data_ptr<float>()),
                num_elements4
            );
        }
        
        // Handle remaining elements
        if (remainder > 0) {
            int threads = 256;
            int blocks = (remainder + threads - 1) / threads;
            softsign_kernel_remainder<<<blocks, threads>>>(
                x.data_ptr<float>(), out.data_ptr<float>(),
                num_elements4 * 4, remainder
            );
        }
    } else {
        // Use multi-stream approach with larger chunk sizes
        const int CHUNK_SIZE = 1 << 22; // 4M elements per chunk
        int num_chunks = (total_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int threads = 512;
        
        const int MAX_STREAMS = 4;
        int num_streams = std::min(num_chunks, MAX_STREAMS);
        std::vector<cudaStream_t> streams(num_streams);
        
        // Create streams with high priority
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, 0);
        }
        
        for (int i = 0; i < num_chunks; i++) {
            int stream_idx = i % num_streams;
            int offset = i * CHUNK_SIZE;
            int current_chunk = std::min(CHUNK_SIZE, total_elements - offset);
            
            // Process chunk elements in vectors of 4
            int num_elements4 = current_chunk / 4;
            int remainder = current_chunk % 4;
            
            if (num_elements4 > 0) {
                int blocks = (num_elements4 + threads - 1) / threads;
                softsign_kernel_shared_vec4<<<blocks, threads, threads * sizeof(float4), streams[stream_idx]>>>(
                    reinterpret_cast<const float4*>(x.data_ptr<float>() + offset),
                    reinterpret_cast<float4*>(out.data_ptr<float>() + offset),
                    num_elements4
                );
            }
            
            if (remainder > 0) {
                int blocks = (remainder + threads - 1) / threads;
                softsign_kernel_remainder<<<blocks, threads, 0, streams[stream_idx]>>>(
                    x.data_ptr<float>(), out.data_ptr<float>(),
                    offset + num_elements4 * 4, remainder
                );
            }
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
    m.def("forward", &forward, "Adaptive Vectorized Softsign activation (CUDA)");
}