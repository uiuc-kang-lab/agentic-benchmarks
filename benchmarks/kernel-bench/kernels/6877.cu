#include <torch/extension.h>
#include <vector>

// CUDA kernel for argmax over a specified dimension using shared memory for performance.
// This kernel assumes the input tensor is contiguous.
//
// x:       the input data
// indices: the output indices (argmax)
// outerSize: the product of all dimensions before 'dim'
// dimSize:   the size of the dimension over which we compute argmax
// innerSize: the product of all dimensions after 'dim'
//
// Each block processes a specific outer_idx. Within each block, threads process inner_idx.
// Shared memory is used to store intermediate max values and indices for reduction within the block.
__global__ void optimized_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize)
{
    extern __shared__ float shared_max_vals[];
    extern __shared__ int shared_max_indices[];

    int inner_idx = threadIdx.x;
    int outer_idx = blockIdx.x;

    if (outer_idx < outerSize) {
        int start_offset = (outer_idx * dimSize * innerSize) + inner_idx;

        float max_val = -FLT_MAX;
        int max_idx = 0;

        // Each thread finds the local max in its segment
        for (int d = 0; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Store local max and index in shared memory
        shared_max_vals[threadIdx.x] = max_val;
        shared_max_indices[threadIdx.x] = max_idx;

        __syncthreads(); // Ensure all threads have written their local max

        // Reduce within block to find the global max
        if (inner_idx == 0) {
            float block_max_val = shared_max_vals[0];
            int block_max_idx = shared_max_indices[0];
            for (int i = 1; i < blockDim.x; i++) {
                if (shared_max_vals[i] > block_max_val) {
                    block_max_val = shared_max_vals[i];
                    block_max_idx = shared_max_indices[i];
                }
            }
            indices[outer_idx] = block_max_idx;
        }
    }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    // Ensure input tensor is float32 (can adapt as needed)
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // We will use contiguous data
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute sizes for outerSize, dimSize, innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // The output shape is the input shape with dimension dim removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    // Create output for indices (type: long)
    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch kernel
    const int threads = 256;
    const int blocks = outerSize;
    size_t shared_memory_size = threads * (sizeof(float) + sizeof(int));

    optimized_argmax_kernel<<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &argmax_forward_cuda, "Optimized ArgMax CUDA forward");
}