#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for frequently accessed, read-only parameters
__constant__ int d_dim_size;
__constant__ int d_stride;

// Kernel that uses constant memory to load dim_size and stride
__global__ void prod_reduce_kernel_const(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int num_elements) {
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int block_offset = blockIdx.x * blockDim.x;

    // Each thread processes multiple output elements
    for (int idx = block_offset + tid; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        // Use constant memory for dim_size and stride
        for (int i = 0; i < d_dim_size; ++i) {
            product *= input[idx + i * d_stride];
        }
        output[idx] = product;
    }
}

// Host function: copy constant parameters to constant memory and launch the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Retrieve sizes and determine output shape
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    // Copy the read-only parameters to constant memory
    cudaMemcpyToSymbol(d_dim_size, &dim_size, sizeof(int));
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Configure with a suitable number of threads and blocks
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel_const<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA) with constant memory optimization");
}
