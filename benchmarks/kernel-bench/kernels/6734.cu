#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* __restrict__ input, 
                                 float* __restrict__ output,
                                 const int dim_size,
                                 const int stride,
                                 const int num_elements,
                                 const int elements_per_thread) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes a fixed number of elements
    const int start_idx = gid * elements_per_thread;
    const int end_idx = min(start_idx + elements_per_thread, num_elements);
    
    // Process elements assigned to this thread
    for (int idx = start_idx; idx < end_idx; idx++) {
        float product = 1.0f;
        #pragma unroll 4
        for (int i = 0; i < dim_size; ++i) {
            product *= input[idx + i * stride];
        }
        output[idx] = product;
    }
    
    // Handle remaining elements
    if (gid == total_threads - 1) {
        for (int idx = total_threads * elements_per_thread; idx < num_elements; idx++) {
            float product = 1.0f;
            for (int i = 0; i < dim_size; ++i) {
                product *= input[idx + i * stride];
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

    // Optimize thread and block count for balanced workload
    const int threads_per_block = 128;
    const int elements_per_thread = 4;
    const int num_threads_needed = (num_elements + elements_per_thread - 1) / elements_per_thread;
    const int num_blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
    
    // Limit number of blocks for better occupancy
    const int max_blocks = 112; // Optimal for H100
    const int actual_blocks = min(num_blocks, max_blocks);

    prod_reduce_kernel<<<actual_blocks, threads_per_block>>>(
        input_ptr, output_ptr, dim_size, stride, num_elements, elements_per_thread);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}