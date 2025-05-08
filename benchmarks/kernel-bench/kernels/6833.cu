#include <torch/extension.h>
#include <vector>
#include <cfloat>

// This kernel minimizes warp divergence by refactoring conditional logic
// into branchless arithmetic using warp shuffle operations.
__global__ void divergence_free_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Determine the output element this block is responsible for
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread computes a local maximum over its assigned elements
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        // Compute a branchless update
        int swap = (val > local_max) ? 1 : 0;
        local_max = swap * val + (1 - swap) * local_max;
        local_idx = swap * i   + (1 - swap) * local_idx;
    }

    // Perform warp-level reduction using __shfl_down_sync in a branch-minimized way
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float candidate = __shfl_down_sync(0xffffffff, local_max, offset);
        int candidate_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        int swap = (candidate > local_max) ? 1 : 0;
        local_max = swap * candidate + (1 - swap) * local_max;
        local_idx = swap * candidate_idx + (1 - swap) * local_idx;
    }

    // Use shared memory to combine results from different warps in the block
    __shared__ float shared_vals[32];
    __shared__ int shared_idxs[32];

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) {
        shared_vals[warpId] = local_max;
        shared_idxs[warpId] = local_idx;
    }
    __syncthreads();

    // Final reduction within the first warp
    if (threadIdx.x < numWarps) {
        float combined = shared_vals[threadIdx.x];
        int combined_idx = shared_idxs[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float candidate = __shfl_down_sync(0xffffffff, combined, offset);
            int candidate_idx = __shfl_down_sync(0xffffffff, combined_idx, offset);
            int swap = (candidate > combined) ? 1 : 0;
            combined = swap * candidate + (1 - swap) * combined;
            combined_idx = swap * candidate_idx + (1 - swap) * combined_idx;
        }

        if (threadIdx.x == 0) {
            indices[global_idx] = combined_idx;
        }
    }
}

// Host function to launch the divergence-free argmax kernel

torch::Tensor divergence_free_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
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

    // Output shape: same as input with the reduced dimension removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) {
            out_sizes.push_back(sizes[d]);
        }
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total_outputs = outerSize * innerSize;
    int blockSize = 128;  // Maintained block size for sufficient parallelism
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    divergence_free_argmax_kernel<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &divergence_free_argmax_forward_cuda, "Divergence Free ArgMax CUDA forward");
}
