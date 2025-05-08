#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel_strided(const float* __restrict__ x, float* __restrict__ y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    extern __shared__ float shared_x[];
    
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Process elements with stride pattern
    while (idx < n) {
        // Cache current batch in shared memory
        shared_x[tid] = x[idx];
        __syncthreads();
        
        // Process element using shared memory
        float xi = shared_x[tid];
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx] = 0.5f * xi * (1.0f + tanh_val);
        
        // Move to next batch with stride
        idx += stride;
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int max_blocks = 256;  // Adjust based on workload
    int blocks = min((n + threads - 1) / threads, max_blocks);
    
    // Allocate shared memory for the block
    size_t shared_mem_size = threads * sizeof(float);
    
    gelu_kernel_strided<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with strided processing");
}