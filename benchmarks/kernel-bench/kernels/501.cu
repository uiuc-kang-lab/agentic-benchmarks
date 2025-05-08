#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

__global__ void streamPipelinedMultiplyKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int64_t chunk_size,
                                             int64_t offset) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < chunk_size) {
        int global_idx = offset + tid;
        // Process 4 elements at a time
        if (tid % 4 == 0 && tid + 3 < chunk_size) {
            float4* a4_ptr = (float4*)(&A[tid]);
            float4* c4_ptr = (float4*)(&C[tid]);
            float4 data = *a4_ptr;
            
            data.x *= s;
            data.y *= s;
            data.z *= s;
            data.w *= s;
            
            *c4_ptr = data;
        }
        // Handle elements that don't fit in float4
        else if (tid >= ((chunk_size / 4) * 4)) {
            C[tid] = A[tid] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s) 
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t total_size = A.numel();
    
    // Create multiple CUDA streams for pipelining
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for each stream
    int64_t chunk_size = (total_size + num_streams - 1) / num_streams;
    chunk_size = ((chunk_size + 255) / 256) * 256; // Round up to multiple of block size
    
    const int threads = 256;
    const int blocks_per_chunk = (chunk_size + threads - 1) / threads;
    
    // Process chunks in different streams
    for (int i = 0; i < num_streams; i++) {
        int64_t offset = i * chunk_size;
        int64_t current_chunk_size = std::min(chunk_size, total_size - offset);
        
        if (current_chunk_size > 0) {
            streamPipelinedMultiplyKernel<<<blocks_per_chunk, threads, 0, streams[i]>>>(
                A.data_ptr<float>() + offset,
                C.data_ptr<float>() + offset,
                s,
                current_chunk_size,
                offset
            );
        }
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream pipelined matrix-scalar multiplication kernel");
}