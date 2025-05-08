#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int dim_size, int stride, int num_elements) {
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int block_offset = blockIdx.x * blockDim.x;
    
    // Each thread handles multiple elements with stride
    for (int idx = block_offset + tid; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        #pragma unroll 4
        for (int i = 0; i < dim_size; ++i) {
            product *= input[idx + i * stride];
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

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Using 128 threads per block for potentially better occupancy
    const int threads = 128;
    // Limit max blocks to number of SMs on H100 (132)
    const int max_blocks = 132;
    const int blocks = std::min((num_elements + threads - 1) / threads, max_blocks);

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}