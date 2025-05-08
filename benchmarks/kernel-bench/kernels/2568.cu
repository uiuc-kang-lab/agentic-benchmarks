#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Configuration constants
const int64_t STREAM_THRESHOLD = 1048576;  // 1M elements
const int NUM_STREAMS = 4;
const int THREADS_PER_BLOCK = 256;
const int VECTOR_SIZE = 4;  // For float4 vectorization

template <typename scalar_t>
__global__ void relu_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size,
    const int64_t offset) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idx = tid + offset;
    
    // Vector-width aligned processing
    if constexpr (std::is_same_v<scalar_t, float>) {
        const int vector_idx = global_idx * VECTOR_SIZE;
        if (vector_idx + VECTOR_SIZE - 1 < size) {
            float4* in4 = (float4*)input;
            float4* out4 = (float4*)output;
            float4 val = in4[tid];
            
            // Process all elements in vector
            #pragma unroll
            for (int i = 0; i < VECTOR_SIZE; i++) {
                reinterpret_cast<float*>(&val)[i] = max(reinterpret_cast<float*>(&val)[i], 0.0f);
            }
            
            out4[tid] = val;
            return;
        }
    }
    
    // Handle non-vector-aligned elements
    if (global_idx < size) {
        const scalar_t val = input[global_idx];
        output[global_idx] = max(val, static_cast<scalar_t>(0));
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t total_size = input.numel();
    
    // Use single kernel launch for small inputs
    if (total_size < STREAM_THRESHOLD) {
        const int vector_elements = total_size / VECTOR_SIZE;
        const int blocks = (vector_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_simple", ([&] {
            relu_kernel_vectorized<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                total_size,
                0
            );
        }));
        return output;
    }
    
    // Use streams for large inputs
    cudaStream_t streams[NUM_STREAMS];
    const int64_t chunk_size = (total_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Create streams with priorities
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    
    #pragma unroll
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priority_high);
    }
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_streamed", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            const int64_t offset = i * chunk_size;
            const int64_t current_chunk_size = min(chunk_size, total_size - offset);
            if (current_chunk_size <= 0) break;
            
            const int vector_elements = current_chunk_size / VECTOR_SIZE;
            const int blocks = (vector_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            
            relu_kernel_vectorized<scalar_t><<<blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                current_chunk_size,
                offset
            );
        }
    }));
    
    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Vectorized ReLU forward (CUDA)");
}