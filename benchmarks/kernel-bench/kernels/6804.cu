#include <torch/extension.h>
#include <vector>

// Using block size of 32 (one warp) for better occupancy
#define BLOCK_SIZE 32

__global__ __launch_bounds__(BLOCK_SIZE)
void warp_aligned_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) 
{
    // Global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = outerSize * innerSize;

    if (idx < total) {
        const int outer_idx = idx / innerSize;
        const int inner_idx = idx % innerSize;
        const int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        float max_val = x[start_offset];
        int max_idx = 0;

        // Perform argmax computation
        #pragma unroll 4
        for (int d = 1; d < dimSize; d++) {
            const float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Write result
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
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

    // Calculate grid size based on smaller block size
    const int total = outerSize * innerSize;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    warp_aligned_argmax_kernel<<<blocks, BLOCK_SIZE>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "Warp-aligned ArgMax CUDA forward");
}