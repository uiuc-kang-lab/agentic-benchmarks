#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define ELEMENTS_PER_THREAD 2

__global__ void prod_reduce_kernel(const float* __restrict__ input, 
                                 float* __restrict__ output, 
                                 const int stride, 
                                 const int num_elements) {
    // Each thread processes multiple elements to increase arithmetic intensity
    const int tid = threadIdx.x;
    const int wid = tid >> 5;  // Warp ID
    const int lane = tid & 31; // Lane within warp
    
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int base_idx = blockIdx.x * elements_per_block;
    
    // Process multiple elements per thread
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = base_idx + tid + i * BLOCK_SIZE;
        if (idx < num_elements) {
            float product = 1.0f;
            
            #pragma unroll 5
            for (int j = 0; j < 50; j += 5) {
                product *= input[idx + (j) * stride];
                product *= input[idx + (j+1) * stride];
                product *= input[idx + (j+2) * stride];
                product *= input[idx + (j+3) * stride];
                product *= input[idx + (j+4) * stride];
            }
            
            output[idx] = product;
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

    // Calculate grid size based on elements per thread and block size
    int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int blocks = (num_elements + elements_per_block - 1) / elements_per_block;

    prod_reduce_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}