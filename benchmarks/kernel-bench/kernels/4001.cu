#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hybrid kernel: grid-stride loop over tiles with shared memory preloading
__global__ void elu_kernel_hybrid(const float* x, float* out, float alpha, int n) {
    extern __shared__ float tile[];  // dynamically allocated shared memory
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int gridSize = gridDim.x;

    // Iterate over tiles using grid-stride loop to cover the full tensor
    for (int tileIdx = blockIdx.x; tileIdx * blockSize < n; tileIdx += gridSize) {
        int index = tileIdx * blockSize + tid;
        
        // Load a tile of elements from global memory into shared memory
        if (index < n) {
            tile[tid] = x[index];
        }
        __syncthreads();
        
        // Process the tile: compute the ELU activation on the preloaded data
        if (index < n) {
            float val = tile[tid];
            out[index] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
        __syncthreads(); // Ensure shared memory is ready for next iteration
    }
}

// Interface function to be called from Python
torch::Tensor elu_cuda_hybrid(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    // Determine number of blocks such that each block processes a tile of 'threads' elements
    int blocks = (n + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float);

    elu_kernel_hybrid<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_hybrid, "Hybrid ELU activation with shared memory and grid-stride loop (CUDA)");
}
