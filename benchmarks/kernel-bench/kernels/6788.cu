#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256

__global__ void prod_reduce_kernel(const float* __restrict__ input, 
                                 float* __restrict__ output, 
                                 const int stride, 
                                 const int num_elements) {
    int idx = blockIdx.x * blockDim.x * 50 + threadIdx.x;
    float product = 1.0f;

    // Loop over the entire workload, blocking by blockDim.x and thread count
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x * 50) {
        float local_product = 1.0f;
        int local_offset = i;
        #pragma unroll
        for (int j = 0; j < 50 && (local_offset + j * stride) < num_elements; ++j) {
            local_product *= input[local_offset + j * stride];
        }
        product *= local_product;
    }

    if (idx < num_elements) {
        output[blockIdx.x] = product;
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
