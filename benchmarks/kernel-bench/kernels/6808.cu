#include <torch/extension.h>
#include <vector>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

// Optimized CUDA kernel to minimize warp divergence
__global__ __launch_bounds__(BLOCK_SIZE)
void warp_optimized_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        // Compute flat index with warp-aligned access
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Initialize comparisons for all threads to ensure warp-aligned access
        float max_val = x[start_offset];
        int max_idx = 0;

        // Avoid branch divergence by ensuring all threads follow same execution flow
        for (int d = 1; d < dimSize; d++) {
            // Use the conditional operator instead of if statement
            float val = x[start_offset + d * innerSize];
            bool update = val > max_val;
            max_val = update ? val : max_val;
            max_idx = update ? d : max_idx;
        }

        // Write the results
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor warp_optimized_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // Make tensor contiguous
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Determine dimensions for the operation
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare output shape (original shape with dim removed)
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Compute grid configuration using the block size
    const int total = outerSize * innerSize;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    warp_optimized_argmax_kernel<<<blocks, BLOCK_SIZE>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_argmax_forward_cuda, "Warp Divergence Optimized ArgMax CUDA forward");
}