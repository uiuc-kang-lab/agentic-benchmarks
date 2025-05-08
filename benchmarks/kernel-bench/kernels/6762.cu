#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    shared[tid] = 1.0f;
    
    // Compute partial products for this thread
    if (idx < num_elements) {
        float local_prod = 1.0f;
        for (int i = 0; i < dim_size; ++i) {
            local_prod *= input[idx + i * stride];
        }
        shared[tid] = local_prod;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && idx + s < num_elements) {
            shared[tid] *= shared[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0 && idx < num_elements) {
        output[blockIdx.x] = shared[0];
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
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    prod_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
        (input_ptr, output_ptr, dim_size, stride, num_elements);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}