#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ int d_dim_size;
__constant__ int d_stride;

__global__ void prod_reduce_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float product = 1.0f;
        for (int i = 0; i < d_dim_size; ++i) {
            product *= input[idx + i * d_stride];
        }
        output[idx] = product;
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

    // Copy constants to device constant memory
    cudaMemcpyToSymbol(d_dim_size, &dim_size, sizeof(int));
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}