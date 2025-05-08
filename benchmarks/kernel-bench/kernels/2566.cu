#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Threshold for using streams (in elements)
const int64_t STREAM_THRESHOLD = 1048576; // 1M elements
const int NUM_STREAMS = 4;

// CUDA kernel for ReLU activation with optimization
template <typename scalar_t>
__global__ void adaptive_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t chunk_size,
    const int64_t offset) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < chunk_size) {
        const int global_idx = idx + offset;
        if constexpr (std::is_same_v<scalar_t, float>) {
            float4* in4 = (float4*)input;
            float4* out4 = (float4*)output;
            if (global_idx % 4 == 0 && global_idx + 3 < chunk_size) {
                float4 val = in4[global_idx/4];
                val.x = val.x > 0 ? val.x : 0;
                val.y = val.y > 0 ? val.y : 0;
                val.z = val.z > 0 ? val.z : 0;
                val.w = val.w > 0 ? val.w : 0;
                out4[global_idx/4] = val;
                return;
            }
        }
        const scalar_t val = input[global_idx];
        output[global_idx] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t total_size = input.numel();
    const int threads = 256;
    
    if (total_size < STREAM_THRESHOLD) {
        const int blocks = (total_size + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "adaptive_relu_kernel", ([&] {
            adaptive_relu_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                total_size,
                0
            );
        }));
        return output;
    }
    
    const int64_t chunk_size = (total_size + NUM_STREAMS - 1) / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priority_high);
    }
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "adaptive_relu_kernel_streamed", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            const int64_t offset = i * chunk_size;
            const int64_t current_chunk_size = min(chunk_size, total_size - offset);
            if (current_chunk_size <= 0) break;
            
            const int blocks = (current_chunk_size + threads - 1) / threads;
            adaptive_relu_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                current_chunk_size,
                offset
            );
        }
    }));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive ReLU forward (CUDA)");
}