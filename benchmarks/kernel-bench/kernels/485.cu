#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel using shared memory and warp-level primitives for reduction
__global__ void sharedMemoryReductionKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int64_t size)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load elements into shared memory
    float val = (idx < size) ? A[idx] * s : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) C[blockIdx.x] = sdata[0];
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty({(A.numel() + 255) / 256}, A.options());
    int64_t size = A.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    sharedMemoryReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(A.data_ptr<float>(),
                                                                              C.data_ptr<float>(),
                                                                              s,
                                                                              size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory reduction kernel");
}