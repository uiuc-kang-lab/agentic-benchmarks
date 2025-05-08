#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void multiplyKernelStreamed(const float* __restrict__ A,
                                        float* __restrict__ C,
                                        float s,
                                        int64_t size)
{
    // Use shared memory for caching
    extern __shared__ float shared_A[];
    
    // Create thread block group
    cg::thread_block block = cg::this_thread_block();
    
    // Calculate indices for processing multiple elements per thread
    const int tid = threadIdx.x;
    const int elements_per_thread = 4;  // Process 4 elements per thread
    const int block_offset = blockIdx.x * blockDim.x * elements_per_thread;
    
    // Load data into shared memory
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = block_offset + i * blockDim.x + tid;
        if (idx < size) {
            shared_A[i * blockDim.x + tid] = A[idx];
        }
    }
    
    // Ensure all threads have loaded their data
    block.sync();
    
    // Process elements
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = block_offset + i * blockDim.x + tid;
        if (idx < size) {
            C[idx] = shared_A[i * blockDim.x + tid] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    multiplyKernelStreamed<<<blocks, threads, 0, stream>>>(A.data_ptr<float>(),
                                                          C.data_ptr<float>(),
                                                          s,
                                                          size);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel with CUDA streams");
}