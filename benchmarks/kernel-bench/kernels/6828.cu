#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Optimized kernel using warp-level primitives for efficient reduction
__global__ void efficient_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Each block computes one output element corresponding to an (outer, inner) pair.
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread computes a local maximum in the reduction dimension
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = x[base_offset + i * innerSize];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp-level reduction using shuffle intrinsics
    // All threads in a warp reduce their values without shared memory
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Use shared memory to reduce across warps
    // Assuming maximum of 1024 threads per block => max 32 warps
    __shared__ float sdata[32];
    __shared__ int sidx[32];

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // First thread of each warp writes its result to shared memory
    if (lane == 0) {
        sdata[warpId] = local_max;
        sidx[warpId] = local_idx;
    }
    __syncthreads();

    // Final reduction: only the first warp participates
    if (threadIdx.x < blockDim.x / warpSize) {
        local_max = sdata[lane];
        local_idx = sidx[lane];
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_max > local_max) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            indices[global_idx] = local_idx;
        }
    }
}

// Host function to launch the efficient argmax kernel

torch::Tensor efficient_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute sizes for outer, dim, and inner dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare output shape: remove the reduced dimension
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total_outputs = outerSize * innerSize;
    int blockSize = 128;  // Chosen to cover reduction dimension; can be tuned for specific scenarios
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    // Launch the kernel. No dynamic shared memory is used because warp reduction minimizes it.
    efficient_argmax_kernel<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_argmax_forward_cuda, "Efficient ArgMax CUDA forward with warp-level reduction");
}
