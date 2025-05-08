#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a stride loop so that each thread can process multiple output elements if needed, 
// which is beneficial when the total number of output elements exceeds the total number of threads available.
// The fixed reduction size of 50 is unrolled for optimal performance.
__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                     int stride, int num_elements) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    
    // Loop over all output elements assigned to this thread
    for (int idx = global_idx; idx < num_elements; idx += grid_stride) {
        float product = 1.0f;
        #pragma unroll
        for (int i = 0; i < 50; ++i) {
            product *= input[idx + i * stride];
        }
        output[idx] = product;
    }
}


torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Extract the sizes and validate the reduction dimension is of size 50
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    TORCH_CHECK(dim_size == 50, "Reduction dimension must have size 50, but got " + std::to_string(dim_size));
    sizes.erase(sizes.begin() + dim);
    
    torch::Tensor output = torch::empty(sizes, x.options());
    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension using stride loops (CUDA)");
}
