#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warpReduceProd(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val *= __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void prod_reduce_kernel(const float* __restrict__ input, 
                                 float* __restrict__ output,
                                 const int dim_size,
                                 const int stride,
                                 const int num_elements) {
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int global_idx = bid * BLOCK_SIZE + tid;
    
    __shared__ float warp_products[BLOCK_SIZE / WARP_SIZE];
    
    // Initialize product
    float thread_prod = 1.0f;
    
    // Compute partial products with vectorized loads where possible
    if (global_idx < num_elements) {
        const float* input_row = input + global_idx;
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            thread_prod *= input_row[i * stride];
        }
    }
    
    // First warp-level reduction
    thread_prod = warpReduceProd(thread_prod);
    
    // Write reduced warp results to shared memory
    if (lane == 0) {
        warp_products[wid] = thread_prod;
    }
    
    __syncthreads();
    
    // Final reduction using first warp
    if (wid == 0) {
        // Load from shared memory
        thread_prod = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_products[lane] : 1.0f;
        
        // Final warp reduction
        thread_prod = warpReduceProd(thread_prod);
        
        // Write result
        if (lane == 0 && bid < (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            output[bid] = thread_prod;
        }
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
    
    const int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    prod_reduce_kernel<<<blocks, BLOCK_SIZE>>>(
        input_ptr,
        output_ptr,
        dim_size,
        stride,
        num_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}