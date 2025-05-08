#include <torch/extension.h>
#include <vector>
#include <float.h>
#include <cuda_fp16.h>

template <typename T>
__device__ void warpReduceMax(T& val, int& idx) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        T tmpVal = __shfl_down_sync(0xffffffff, val, offset);
        int tmpIdx = __shfl_down_sync(0xffffffff, idx, offset);
        if (tmpVal > val || (tmpVal == val && tmpIdx < idx)) {
            val = tmpVal;
            idx = tmpIdx;
        }
    }
}

__global__ void argmax_kernel_coop_opt(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block handles multiple slices for better occupancy
    int slice = blockIdx.x * blockDim.y + threadIdx.y;
    if (slice >= outerSize * innerSize) return;

    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;
    int base_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    // Per-thread reduction
    float local_max = -FLT_MAX;
    int local_argmax = 0;

    // Coalesced memory access pattern
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float curr_val = x[base_offset + d * innerSize];
        if (curr_val > local_max) {
            local_max = curr_val;
            local_argmax = d;
        }
    }

    // Warp-level reduction
    float warp_max = local_max;
    int warp_argmax = local_argmax;
    warpReduceMax(warp_max, warp_argmax);

    // Cross-warp reduction
    __shared__ float smax[32];
    __shared__ int sidx[32];

    if (threadIdx.x % warpSize == 0) {
        smax[threadIdx.x / warpSize] = warp_max;
        sidx[threadIdx.x / warpSize] = warp_argmax;
    }
    __syncthreads();

    // First warp reduces final results
    if (threadIdx.x < warpSize) {
        float val = (threadIdx.x < blockDim.x / warpSize) ? smax[threadIdx.x] : -FLT_MAX;
        int idx = (threadIdx.x < blockDim.x / warpSize) ? sidx[threadIdx.x] : 0;
        warpReduceMax(val, idx);
        
        if (threadIdx.x == 0) {
            indices[slice] = idx;
        }
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim");

    // Calculate dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    // Prepare output
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if (d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    // Optimized launch config
    int slices = outerSize * innerSize;
    dim3 block(128, 4);  // x=reduction threads, y=slices per block
    dim3 grid((slices + block.y - 1) / block.y);

    argmax_kernel_coop_opt<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA with optimized cooperative reduction");
}