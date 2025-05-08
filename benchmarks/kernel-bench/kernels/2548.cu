#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int NUM_STREAMS = 4;

template <typename scalar_t>
__global__ void relu_kernel_streamed(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t chunk_size,
    const int64_t offset) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < chunk_size) {
        const int global_idx = idx + offset;
        const scalar_t val = input[global_idx];
        output[global_idx] = val > 0 ? val : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int64_t total_size = input.numel();
    const int64_t chunk_size = (total_size + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_streamed", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            const int64_t offset = i * chunk_size;
            const int64_t current_chunk_size = min(chunk_size, total_size - offset);
            if (current_chunk_size <= 0) break;
            
            const int blocks = (current_chunk_size + threads - 1) / threads;
            
            relu_kernel_streamed<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                current_chunk_size,
                offset
            );
        }
    }));
    
    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with stream pipelining (CUDA)");
}