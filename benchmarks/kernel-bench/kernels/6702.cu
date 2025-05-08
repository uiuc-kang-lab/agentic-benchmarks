#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define block size for thread reduction
#define BLOCK_SIZE 256

// Kernel: Each block computes one output element reduction over the specified dimension.
// The grid is defined in 2D (blockIdx.x, blockIdx.y) to map to output elements more efficiently
__global__ void multidim_indexed_prod_reduction_kernel(const float* __restrict__ input,
                                                         float* __restrict__ output,
                                                         int dim_size,
                                                         int stride,
                                                         int total_output) {
    // Compute global output index using 2D grid indexing
    int outIdx = blockIdx.y * gridDim.x + blockIdx.x;
    if (outIdx >= total_output) return;

    float prod = 1.0f;
    // Each thread processes a subset of the reduction dimension
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        prod *= input[outIdx + i * stride];
    }

    // Reduce partial products within the block using shared memory
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = prod;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] *= sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread writes the final product to the output
    if (threadIdx.x == 0) {
        output[outIdx] = sdata[0];
    }
}

// Forward function for product reduction over a specified dimension using multi-dimensional grid indexing
// This mapping improves the distribution of work across thread blocks for better occupancy

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    // Determine output shape by removing the reduction dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());
    
    int total_output = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Compute 2D grid dimensions to map each output element to a block
    int grid_x = static_cast<int>(sqrtf(total_output));
    if (grid_x < 1) grid_x = 1;
    int grid_y = (total_output + grid_x - 1) / grid_x;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_SIZE);
    
    multidim_indexed_prod_reduction_kernel<<<grid, block>>>(input_ptr, output_ptr, dim_size, stride, total_output);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Multi-dim indexed product reduction (CUDA)");
}
