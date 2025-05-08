#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Store the constant value 1.0f in constant memory
__constant__ float c_one = 1.0f;

__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Grid-stride loop for better utilization of GPU resources
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float val = x[i];
        // Use constant memory for the value 1.0f
        float sigmoid = __fdividef(c_one, c_one + expf(-val));
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    
    // Split the work into multiple streams for concurrent execution
    constexpr int num_streams = 4;  // Adjust based on GPU capabilities
    const int elements_per_stream = (n + num_streams - 1) / num_streams;
    
    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Launch kernels in different streams
    for (int i = 0; i < num_streams; i++) {
        int64_t stream_offset = i * elements_per_stream;
        int64_t stream_elements = min(elements_per_stream, n - stream_offset);
        if (stream_elements <= 0) break;
        
        const int stream_blocks = (stream_elements + threads - 1) / threads;
        optimized_swish_kernel<<<stream_blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>() + stream_offset,
            y.data_ptr<float>() + stream_offset,
            stream_elements
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation forward pass (CUDA)");
}