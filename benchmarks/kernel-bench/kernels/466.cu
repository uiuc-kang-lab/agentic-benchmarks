#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size,
                               int64_t offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[offset + idx] = A[offset + idx] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int num_streams = 4;
    const int chunk_size = (size + num_streams - 1) / num_streams;
    const int threads = 256;
    
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    for (int i = 0; i < num_streams; i++) {
        int64_t offset = i * chunk_size;
        int64_t current_size = min(chunk_size, size - offset);
        if (current_size <= 0) break;
        
        const int blocks = (current_size + threads - 1) / threads;
        
        multiplyKernel<<<blocks, threads, 0, streams[i]>>>
            (A.data_ptr<float>(),
             C.data_ptr<float>(),
             s,
             current_size,
             offset);
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}