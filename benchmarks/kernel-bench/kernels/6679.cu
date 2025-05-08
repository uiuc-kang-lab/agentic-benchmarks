#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// This kernel parallelizes the product reduction over the specified dimension by using multiple threads per output element.
// Each block is responsible for computing one output element. Threads in the block compute partial products over a strided
// subset of the reduction dimension, accumulate their partial results in shared memory, and then perform a tree-based reduction.

__global__ void prod_reduce_kernel_parallel(const float* input, float* output, int dim_size, int stride) {
    // Each block computes one output element
    int out_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Each thread computes a partial product over the reduction dimension, striding by the blockDim
    float partial = 1.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        partial *= input[out_idx + i * stride];
    }
    sdata[tid] = partial;
    __syncthreads();

    // Perform reduction in shared memory (product reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads(); // Synchronize only when shared memory values are updated
    }

    // The first thread in the block writes the final product to the output
    if (tid == 0) {
        output[out_idx] = sdata[0];
    }
}


// Forward function: computes product reduction over a specified dimension.
// It reshapes the output tensor by removing the reduced dimension and launches one block per output element.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Get the input sizes and remove the reduced dimension to form the output shape
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch configuration: one block per output element with 256 threads per block.
    // Dynamic shared memory allocation is used to hold 256 floats per block.
    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    prod_reduce_kernel_parallel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension (CUDA)");
}
