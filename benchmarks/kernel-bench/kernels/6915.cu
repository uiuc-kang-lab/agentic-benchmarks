#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// This kernel computes argmax over a specified dimension using only warp-level primitives.
// Each block is assigned one (outer, inner) pair and is launched with exactly 32 threads (one warp).
// Each thread processes several elements along the reduction dimension in a stride loop, utilizing shared memory for improved performance.
// Then, warp-level intrinsic __shfl_down_sync() is used to reduce and determine the maximum value and its index,
// completely avoiding shared memory operations for the reduction phase.

__global__ void warp_argmax_nosm_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block handles one (outer, inner) pair
    int idx = blockIdx.x;
    if (idx >= outerSize * innerSize) return;

    int outer_idx = idx / innerSize;
    int inner_idx = idx % innerSize;
    int start_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    // Each thread in the warp computes a partial maximum over the reduction dimension.
    // Using a stride loop with a step equal to the warp size.
    float thread_max = -FLT_MAX;
    int thread_arg = 0;
    const int warpSize = 32;

    for (int d = threadIdx.x; d < dimSize; d += warpSize) {
        // Use __ldg to enable read-only cache and improved performance
        float val = __ldg(&x[start_offset + d * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_arg = d;
        } else if (val == thread_max && d < thread_arg) {
            // Tie-breaker: choose the smaller index
            thread_arg = d;
        }
    }

    // Perform warp-level reduction using shuffle intrinsics
    unsigned int mask = 0xffffffff; // Full mask for 32 threads
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, thread_max, offset);
        int other_arg = __shfl_down_sync(mask, thread_arg, offset);
        if (other_max > thread_max) {
            thread_max = other_max;
            thread_arg = other_arg;
        } else if (other_max == thread_max && other_arg < thread_arg) {
            thread_arg = other_arg;
        }
    }

    // The first thread in the warp writes the final argmax result
    if (threadIdx.x == 0) {
        indices[idx] = thread_arg;
    }
}

// Host function to launch the CUDA kernel for argmax
// This function computes outerSize, dimSize, and innerSize based on the input tensor dimensions
// and then launches one warp (32 threads) per (outer, inner) pair.

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    const int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    int outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }

    // Build the output shape by removing the reduction dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_sizes.push_back(sizes[i]);
        }
    }
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each output element corresponds to one outer*inner pair
    int total = outerSize * innerSize;
    // Launch one warp (32 threads) per output element
    const int threads = 32;
    const int blocks = total;

    warp_argmax_nosm_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize);

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (warp-level reduction, no shared memory)");
}
