#include <torch/extension.h>
#include <vector>

// Optimized CUDA kernel using 2D thread mapping
// Each thread now directly computes for an (outer, inner) pair
__global__ void argmax_kernel_2d(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {
    // Map the two dimensions: blockIdx.y*blockDim.y + threadIdx.y -> outer index, and 
    // blockIdx.x*blockDim.x + threadIdx.x -> inner index
    int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outerSize && inner_idx < innerSize) {
        int start_offset = outer_idx * (dimSize * innerSize) + inner_idx;
        float max_val = x[start_offset];
        int max_idx = 0;

        // Iterate along the dimension to compute argmax
        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Store the computed index
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to launch the CUDA kernel with 2D grid configuration
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute dimensions: outerSize, dimSize, and innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Output tensor shape: remove the argmax dimension
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Configure block dimensions optimally for 2D mapping
    dim3 block(32, 8);
    dim3 grid((innerSize + block.x - 1) / block.x,
              (outerSize + block.y - 1) / block.y);

    argmax_kernel_2d<<<grid, block>>>(
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
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with 2D thread mapping");
}
