#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int stride, int num_elements) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // Each thread computes its local product
    float thread_product = 1.0f;
    if (idx < num_elements) {
        #pragma unroll
        for (int i = 0; i < 50; i += ELEMENTS_PER_THREAD) {
            float temp_prod = 1.0f;
            #pragma unroll
            for (int j = 0; j < ELEMENTS_PER_THREAD && (i + j) < 50; ++j) {
                temp_prod *= input[idx + (i + j) * stride];
            }
            thread_product *= temp_prod;
        }
    }
    
    // Store in shared memory
    shared_data[tid] = thread_product;
    __syncthreads();
    
    // First thread in block writes final result
    if (tid == 0 && idx < num_elements) {
        output[blockIdx.x] = thread_product;
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

    int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    prod_reduce_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}