#include <torch/extension.h>

__global__ void unrolled_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int TILE_SIZE = 1024; // Size of shared memory tile
    const int elements_per_thread = 4;
    
    __shared__ float s_data[TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * TILE_SIZE;
    const int grid_stride = gridDim.x * TILE_SIZE;
    
    // Process input array with grid stride loop using shared memory tiles
    for (int tile_base = block_offset; tile_base < n; tile_base += grid_stride) {
        // Load tile into shared memory
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tile_base + tid * elements_per_thread + i;
            if (idx < n) {
                s_data[tid * elements_per_thread + i] = x[idx];
            }
        }
        
        __syncthreads();
        
        // Process data in shared memory
        float vals[4];
        float results[4];
        
        // Load from shared memory
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tid * elements_per_thread + i;
            vals[i] = (tile_base + idx < n) ? s_data[idx] : 0.0f;
        }
        
        // Compute swish
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            float sigmoid = __fdividef(1.0f, 1.0f + expf(-vals[i]));
            results[i] = vals[i] * sigmoid;
        }
        
        __syncthreads();
        
        // Store results back to global memory
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tile_base + tid * elements_per_thread + i;
            if (idx < n) {
                y[idx] = results[i];
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    unrolled_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with 4x unrolling (CUDA)");
}