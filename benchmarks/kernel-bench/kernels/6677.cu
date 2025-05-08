#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ void warpProd(volatile float* sdata, int tid) {
    if (BLOCK_SIZE >= 64) sdata[tid] *= sdata[tid + 32];
    if (BLOCK_SIZE >= 32) sdata[tid] *= sdata[tid + 16];
    if (BLOCK_SIZE >= 16) sdata[tid] *= sdata[tid + 8];
    if (BLOCK_SIZE >= 8) sdata[tid] *= sdata[tid + 4];
    if (BLOCK_SIZE >= 4) sdata[tid] *= sdata[tid + 2];
    if (BLOCK_SIZE >= 2) sdata[tid] *= sdata[tid + 1];
}

__global__ void prod_reduce_kernel(const float* input, float* output, 
                                 int dim_size, int stride, int num_elements) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    float thread_prod = 1.0f;
    
    // Each thread processes multiple elements
    if (global_idx < num_elements) {
        for (int i = 0; i < dim_size; i++) {
            thread_prod *= input[global_idx + i * stride];
        }
    }
    
    sdata[tid] = thread_prod;
    __syncthreads();
    
    // Reduce within warps
    if (tid < WARP_SIZE) {
        warpProd(sdata, tid);
    }
    __syncthreads();
    
    // Final reduction and write to output
    if (tid == 0 && global_idx < num_elements) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());
    
    int num_elements = output.numel();
    int stride = x.stride(dim);
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int threads = BLOCK_SIZE;
    int blocks = (num_elements + threads - 1) / threads;
    
    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, 
                                          dim_size, stride, num_elements);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}