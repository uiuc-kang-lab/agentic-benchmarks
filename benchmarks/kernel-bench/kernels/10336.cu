#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants stored in constant memory for faster access
__constant__ float SQRT_2_OVER_PI = 0.7978845608f;
__constant__ float COEFF = 0.044715f;

template<int BLOCK_SIZE = 256, int ITEMS_PER_THREAD = 4>
__global__ void gelu_kernel_fused(const float* __restrict__ x, 
                                 float* __restrict__ y, 
                                 const int n) {
    // Shared memory declaration
    extern __shared__ float shared_x[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD + tid;
    
    // Vectorized loading into shared memory
    float4* shared_vec = reinterpret_cast<float4*>(shared_x);
    float4* input_vec = reinterpret_cast<float4*>(const_cast<float*>(x));
    
    if (gid < n) {
        const int vec_idx = tid;
        const int global_vec_idx = blockIdx.x * (BLOCK_SIZE) + vec_idx;
        
        if (global_vec_idx * 4 + 3 < n) {
            shared_vec[vec_idx] = input_vec[global_vec_idx];
        } else {
            // Handle edge cases manually
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                const int idx = gid + i * BLOCK_SIZE;
                if (idx < n) {
                    shared_x[tid + i * BLOCK_SIZE] = x[idx];
                }
            }
        }
    }
    __syncthreads();
    
    // Process elements using shared memory with loop unrolling
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = gid + i * BLOCK_SIZE;
        if (idx < n) {
            const float xi = shared_x[tid + i * BLOCK_SIZE];
            const float xi2 = xi * xi;
            const float x_cubed = xi2 * xi;
            const float inner = fmaf(COEFF, x_cubed, xi);  // Uses FMA instruction
            const float scaled_inner = inner * SQRT_2_OVER_PI;
            const float tanh_val = tanhf(scaled_inner);
            y[idx] = 0.5f * xi * (1.0f + tanh_val);
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int n = x.numel();
    
    constexpr int BLOCK_SIZE = 256;
    constexpr int ITEMS_PER_THREAD = 4;
    const int blocks = (n + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    
    const size_t shared_mem_size = BLOCK_SIZE * ITEMS_PER_THREAD * sizeof(float);
    
    gelu_kernel_fused<BLOCK_SIZE, ITEMS_PER_THREAD><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward CUDA implementation");
}