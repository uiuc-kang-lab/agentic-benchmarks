#include <torch/extension.h>
#include <vector>

// Templated kernel with BLOCK_SIZE as a compile-time parameter
template <int BLOCK_SIZE>
__global__ void argmax_kernel_tuned(const float* __restrict__ x,
                                      int64_t* __restrict__ indices,
                                      const int outerSize,
                                      const int dimSize,
                                      const int innerSize) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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

// Host function which computes grid configuration and experiments with block sizes
// The optimal block size is chosen based on the total workload size
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    
    // Ensure the input tensor is contiguous
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize, dimSize, and innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Total number of (outer, inner) pairs
    int total = outerSize * innerSize;

    // Experiment with block sizes based on the workload: choose from 32, 64, 128, 256, 512
    int block_size = 256;  // default
    if (total <= 1024) {
        block_size = 32;
    } else if (total <= 8192) {
        block_size = 64;
    } else if (total <= 32768) {
        block_size = 128;
    } else if (total <= 131072) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    int blocks = (total + block_size - 1) / block_size;

    // Prepare output shape (input shape with the specified dimension removed)
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch kernel based on the selected block size using templated instantiation
    switch(block_size) {
        case 32:
            argmax_kernel_tuned<32><<<blocks, 32>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 64:
            argmax_kernel_tuned<64><<<blocks, 64>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 128:
            argmax_kernel_tuned<128><<<blocks, 128>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 256:
            argmax_kernel_tuned<256><<<blocks, 256>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 512:
            argmax_kernel_tuned<512><<<blocks, 512>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        default:
            // Fallback
            argmax_kernel_tuned<256><<<blocks, 256>>>(
                x_contig.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
    }

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with tunable block size");
}
