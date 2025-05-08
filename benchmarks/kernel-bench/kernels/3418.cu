#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

// Kernel to process a contiguous chunk of the input tensor.
// Each thread processes one element in the chunk.
__global__ void gelu_kernel_chunk(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    size_t offset,
                                    size_t chunk_size,
                                    size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_index = offset + idx;
    if (global_index < offset + chunk_size && global_index < numel) {
        float val = input[global_index];
        // GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
        output[global_index] = val * 0.5f * (1.0f + erff(val / 1.4142135623730951f));
    }
}

// Forward function that partitions the work into chunks and uses CUDA streams
// to overlap kernel execution with (potential) memory transfers.
// This approach pipelines computation across multiple streams.

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "Only float32 is supported");

    auto output = torch::empty_like(x);
    const size_t numel = x.numel();

    // Decide the number of streams to use based on tensor size.
    int num_streams = 4; // Tunable parameter
    if (numel < (1 << 14)) { // For small tensors, use the default stream
        num_streams = 1;
    }

    // Partition the total number of elements into roughly equal chunks
    size_t chunk_size = (numel + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaError_t err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        TORCH_CHECK(err == cudaSuccess, "Failed to create CUDA stream");
    }

    const int threads = 256;
    
    // Launch the kernel on each chunk using its own stream.
    for (int i = 0; i < num_streams; i++) {
        size_t offset = i * chunk_size;
        if (offset >= numel) break;
        size_t current_chunk = std::min(chunk_size, numel - offset);
        int blocks = (current_chunk + threads - 1) / threads;
        
        gelu_kernel_chunk<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            offset,
            current_chunk,
            numel
        );
    }

    // Synchronize all created streams and destroy them
    for (int i = 0; i < num_streams; i++) {
        cudaError_t err = cudaStreamSynchronize(streams[i]);
        TORCH_CHECK(err == cudaSuccess, "Stream synchronization failed: ", cudaGetErrorString(err));
        cudaStreamDestroy(streams[i]);
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with stream pipelining");
}
