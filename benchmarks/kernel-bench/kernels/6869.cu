#include <torch/extension.h>
#include <vector>
#include <float.h>

__global__ void argmax_kernel_warp_prim(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    int slice = blockIdx.x;
    if (slice >= outerSize * innerSize) return;

    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;
    int base_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    float local_max = -FLT_MAX;
    int local_argmax = 0;

    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float curr_val = x[base_offset + d * innerSize];
        if (curr_val > local_max) {
            local_max = curr_val;
            local_argmax = d;
        }
    }

    // Use warp-level primitives for reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float max_shuffled = __shfl_down_sync(0xffffffff, local_max, offset);
        int idx_shuffled = __shfl_down_sync(0xffffffff, local_argmax, offset);
        if (max_shuffled > local_max) {
            local_max = max_shuffled;
            local_argmax = idx_shuffled;
        }
    }

    if (threadIdx.x % warpSize == 0) {
        atomicMax(&indices[slice], local_argmax);
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
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
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int slices = outerSize * innerSize;

    const int threads = 128;
    int blocks = slices;

    argmax_kernel_warp_prim<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with warp primitives");
}
