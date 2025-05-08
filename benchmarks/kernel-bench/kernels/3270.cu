#include <torch/extension.h>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, const int64_t n) {
    extern __shared__ float shared_x[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = block_size * gridDim.x;
    
    // Process multiple chunks per block
    for (int base = bid * block_size; base < n; base += grid_size) {
        const int idx = base + tid;
        
        // Load input data into shared memory
        if (idx < n) {
            shared_x[tid] = x[idx];
        }
        __syncthreads();
        
        // Process data from shared memory
        if (idx < n) {
            const float val = shared_x[tid];
            const float sigmoid = 1.0f / (1.0f + expf(-val));
            y[idx] = val * sigmoid;
        }
        __syncthreads();
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    swish_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with shared memory (CUDA)");
}