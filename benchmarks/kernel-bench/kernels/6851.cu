#include <torch/extension.h>
#include <vector>

// Constant memory for dimension sizes
__constant__ int c_outerSize;
__constant__ int c_dimSize;
__constant__ int c_innerSize;

__global__ void argmax_kernel_const(
    const float* __restrict__ x,
    int64_t* __restrict__ indices) {
    // Use constant memory values directly
    int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < c_outerSize && inner_idx < c_innerSize) {
        int start_offset = outer_idx * (c_dimSize * c_innerSize) + inner_idx;
        float max_val = x[start_offset];
        int max_idx = 0;

        // Use constant memory for loop bound
        #pragma unroll 32
        for (int d = 1; d < c_dimSize; d++) {
            float val = x[start_offset + d * c_innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        indices[outer_idx * c_innerSize + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Calculate dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Copy dimension sizes to constant memory
    cudaMemcpyToSymbolAsync(c_outerSize, &outerSize, sizeof(int), 0);
    cudaMemcpyToSymbol(c_dimSize, &dimSize, sizeof(int));
    cudaMemcpyToSymbol(c_innerSize, &innerSize, sizeof(int));

    // Prepare output tensor
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch kernel with 2D grid
    dim3 block(32, 8);
    dim3 grid((innerSize + block.x - 1) / block.x,
              (outerSize + block.y - 1) / block.y);

    argmax_kernel_const<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>()
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with constant memory");
}