#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Use constant memory for the small, read-only parameters
__constant__ int d_dim_size;
__constant__ int d_stride;

// Optimized kernel using constant memory for parameters and loop unrolling
__global__ void prod_reduce_kernel_combined(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int num_elements) {
    const int tid = threadIdx.x;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    const int total_threads = blockDim.x * gridDim.x;

    // Each thread processes multiple output elements
    for (int idx = global_thread_id; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        // Unroll the inner loop for better performance if d_dim_size is small
        const int dim = d_dim_size;
const int stride = d_stride;

// Use pointer arithmetic to avoid repeated multiplication
const float* ptr = input + idx;
#pragma unroll
for (int i = 0; i < dim; ++i) {
    product *= *ptr;
    ptr += stride;
}
        output[idx] = product;
    }
}

// Host function to prepare the tensor and launch the optimized kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Get output sizes by removing the reduced dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);

    // Create output tensor
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    // Copy frequently accessed, read-only parameters to constant memory
    cudaMemcpyToSymbol(d_dim_size, &dim_size, sizeof(int));
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel_combined<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension using constant memory and loop unrolling (CUDA)");
}
