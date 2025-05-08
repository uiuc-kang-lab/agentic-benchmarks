#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// CUDA kernel for argmax over a specified dimension using warp-level shuffle reduction
// in a branchless manner to minimize warp divergence. Each block handles one (outer, inner) pair.
// Every thread in the block computes a partial maximum and its index across a subset of the "dim" dimension,
// then warp shuffle reduction is used to combine these, followed by a shared memory reduction across warps.

__global__ void argmax_shuffle_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block processes one combination of (outer, inner)
    const int k = blockIdx.x;
    if (k >= outerSize * innerSize) return;

    const int outer_idx = k / innerSize;
    const int inner_idx = k % innerSize;
    const int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread processes a subset of the dim dimension
    float local_max = -CUDART_INF_F; // Initialize to negative infinity
    int local_arg = 0;

    // Stride loop across the dimension
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = x[start_offset + d * innerSize];
        // Use a branchless selection via ternary operator
        bool cond = (val > local_max) || ((val == local_max) && (d < local_arg));
        local_max = cond ? val : local_max;
        local_arg = cond ? d   : local_arg;
    }

    // Warp-level reduction using shuffle intrinsics in a branchless way
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, local_max, offset);
        int   other_arg = __shfl_down_sync(mask, local_arg, offset);
        bool cond = (other_max > local_max) || ((other_max == local_max) && (other_arg < local_arg));
        local_max = cond ? other_max : local_max;
        local_arg = cond ? other_arg : local_arg;
    }

    // Allocate shared memory to collect warp-level results
    __shared__ float shared_max[32];  // Maximum possible warps per block
    __shared__ int   shared_arg[32];

    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;  // thread index divided by warpSize

    // Write the result of each warp's reduction to shared memory
    if (lane == 0) {
        shared_max[warpId] = local_max;
        shared_arg[warpId] = local_arg;
    }
    __syncthreads();

    // Final reduction: the first warp reduces the results of all warps
    if (warpId == 0) {
        // Load the warp results; only the first ceil(blockDim.x/32) lanes are valid
        int warp_count = (blockDim.x + 31) / 32;
        local_max = (lane < warp_count) ? shared_max[lane] : -CUDART_INF_F;
        local_arg = (lane < warp_count) ? shared_arg[lane] : 0;
        
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(mask, local_max, offset);
            int   other_arg = __shfl_down_sync(mask, local_arg, offset);
            bool cond = (other_max > local_max) || ((other_max == local_max) && (other_arg < local_arg));
            local_max = cond ? other_max : local_max;
            local_arg = cond ? other_arg : local_arg;
        }
        
        if (lane == 0) {
            indices[k] = local_arg;
        }
    }
}

// Host function to launch the CUDA kernel for argmax
// Computes argmax over dimension 'dim' of the input tensor 'x'.
// The input tensor is assumed to be contiguous and of type float.

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize, dimSize, innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // The output shape is the original shape with dimension 'dim' removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) {
            out_sizes.push_back(sizes[d]);
        }
    }

    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each block handles one (outer, inner) combination
    const int blocks = outerSize * innerSize;
    const int threads = 256;

    argmax_shuffle_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (shuffle reduction, minimized divergence)");
}
