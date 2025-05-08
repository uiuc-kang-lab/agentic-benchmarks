#include <torch/extension.h>
#include <vector>

// Define a function to return the best block size based on problem size and empirical tests
__host__ int compute_optimal_block_size(int total_elements) {
    // Heuristic or empirical rules can be more sophisticated based on architecture testing
    if (total_elements < 1024) {
        return 32;
    } else if (total_elements < 4096) {
        return 64;
    } else if (total_elements < 16384) {
        return 128;
    } else if (total_elements < 65536) {
        return 256;
    } else {
        return 512;
    }
}

// Optimized CUDA kernel with a tunable block size using __launch_bounds__ for additional optimization.
__global__ __launch_bounds__(512) // Use the maximum block size for launch bounds
void adaptive_block_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        float max_val = x[start_offset];
        int max_idx = 0;

        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to determine and use optimal block size
torch::Tensor adaptive_block_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total = outerSize * innerSize;
    int block_size = compute_optimal_block_size(total);
    int blocks = (total + block_size - 1) / block_size;

    adaptive_block_argmax_kernel<<<blocks, block_size>>>(
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
    m.def("forward", &adaptive_block_argmax_forward_cuda, "Adaptive ArgMax CUDA forward");
}
