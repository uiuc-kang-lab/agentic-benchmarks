#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Kernel using warp-level primitives for reduction
__global__ void warp_primitive_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    // Use warp-level primitives for reduction
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float max_val = __shfl_down_sync(0xFFFFFFFF, thread_max, offset);
        int max_idx = __shfl_down_sync(0xFFFFFFFF, thread_max_idx, offset);
        if (max_val > thread_max) {
            thread_max = max_val;
            thread_max_idx = max_idx;
        }
    }

    // Write the result from the first thread of each warp
    if (threadIdx.x % warpSize == 0) {
        atomicMax(&indices[global_idx], thread_max_idx);
    }
}

torch::Tensor warp_primitive_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if (d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    int blockSize = 256;
    dim3 grid(outerSize * innerSize);

    warp_primitive_argmax_kernel<<<grid, blockSize>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_primitive_argmax_forward_cuda, "Warp Primitive ArgMax CUDA forward");
}