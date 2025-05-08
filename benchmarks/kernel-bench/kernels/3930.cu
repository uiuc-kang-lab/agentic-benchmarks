#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void softsign_kernel_shared(const float* __restrict__ x, 
                                     float* __restrict__ out, 
                                     const int num_elements) {
    __shared__ float shared_data[BLOCK_SIZE * ELEMENTS_PER_THREAD + BLOCK_SIZE]; // Extra space for potential overlap
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;
    
    // Each thread loads multiple elements into shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * BLOCK_SIZE;
        if (idx < num_elements) {
            shared_data[tid + i * BLOCK_SIZE] = x[idx];
        }
    }
    
    __syncthreads();
    
    // Process elements from shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * BLOCK_SIZE;
        if (idx < num_elements) {
            float val = shared_data[tid + i * BLOCK_SIZE];
            out[idx] = val / (1.0f + fabsf(val));
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    
    softsign_kernel_shared<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with shared memory (CUDA)");
}