#include <torch/extension.h>
#include <vector>

// Define block size as a compile-time constant to be tuned.
// You can experiment with different values: 32, 64, 128, 256, 512
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Optimized CUDA kernel with a tunable block size using __launch_bounds__ for additional optimization.
__global__ __launch_bounds__(BLOCK_SIZE)
void tuned_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        // Determine outer and inner indices
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        float max_val = x[start_offset];
        int max_idx = 0;

        // Perform a simple linear scan along the target dimension
        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Write the result
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to launch the tuned CUDA kernel
torch::Tensor tuned_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    // Check that input is float32
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // Make the tensor contiguous
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute dimensions for the operation
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

    // Compute grid configuration using the tuned block size
    const int total = outerSize * innerSize;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    tuned_argmax_kernel<<<blocks, BLOCK_SIZE>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tuned_argmax_forward_cuda, "Tuned ArgMax CUDA forward");
}
