#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel that leverages shared memory to stage a tile of the input before processing
// Each block loads a tile of input elements into shared memory, multiplies by the scalar in shared memory, and writes back to global memory

__global__ void sharedMemMultiplyKernel(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int n) {
    extern __shared__ float tile[];  // dynamically allocated shared memory
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load from global memory into shared memory
    if (globalIdx < n) {
        tile[localIdx] = A[globalIdx];
    }
    __syncthreads();

    // Perform multiplication in shared memory
    if (globalIdx < n) {
        tile[localIdx] *= s;
    }
    __syncthreads();

    // Write the result from shared memory back to global memory
    if (globalIdx < n) {
        C[globalIdx] = tile[localIdx];
    }
}

// Forward function to setup kernel execution

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int n = A.numel();
    const int threads = 256;
    
    // Split the work into multiple streams for concurrent execution
    const int num_streams = 4;  // Use 4 CUDA streams
    const int elements_per_stream = (n + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    for (int i = 0; i < num_streams; i++) {
        int stream_offset = i * elements_per_stream;
        int stream_elements = std::min(elements_per_stream, n - stream_offset);
        if (stream_elements <= 0) break;
        
        const int stream_blocks = (stream_elements + threads - 1) / threads;
        size_t sharedMemSize = threads * sizeof(float);
        
        sharedMemMultiplyKernel<<<stream_blocks, threads, sharedMemSize, streams[i]>>>(
            A.data_ptr<float>() + stream_offset,
            C.data_ptr<float>() + stream_offset,
            s,
            stream_elements
        );
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel using shared memory");
}
