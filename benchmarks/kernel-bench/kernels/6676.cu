#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    extern __shared__ float shared_prod[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float thread_prod = 1.0f;
    
    // Compute partial products for this thread
    if (idx < num_elements) {
        for (int i = 0; i < dim_size; ++i) {
            thread_prod *= input[idx + i * stride];
        }
    }
    
    shared_prod[tid] = thread_prod;
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int s = blockDim.x/2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            shared_prod[tid] *= shared_prod[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using warp-level primitives
    if (tid < WARP_SIZE) {
        float warp_prod = shared_prod[tid];
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }
        
        if (tid == 0 && blockIdx.x * blockDim.x < num_elements) {
            output[blockIdx.x] = warp_prod;
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
    
    int threads = 256; // Reduced thread count for better occupancy
    int blocks = (num_elements + threads - 1) / threads;
    int shared_mem_size = threads * sizeof(float);
    
    prod_reduce_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride, num_elements);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}