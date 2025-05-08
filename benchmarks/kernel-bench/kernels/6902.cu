#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// This kernel uses warp-level primitives to perform the reduction without explicit shared memory.
// Each block processes one output element (one outer and inner index pair) and is launched with exactly 32 threads (one warp).
// Each thread iterates over the reduction dimension with a stride of 32 and then a warp-level reduction using __shfl_down_sync
// computes the maximum value and corresponding index. The first thread in the warp writes the result to the output tensor.

__global__ void warp_argmax_kernel(
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
    int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread in the warp computes a partial maximum over the reduction dimension
    float thread_max = -FLT_MAX;
    int thread_arg = 0;

    // Loop over the reduction dimension with a stride equal to the warp size (32)
    for (int d = threadIdx.x; d < dimSize; d += 32) {
        float val = __ldg(&x[start_offset + d * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_arg = d;
        }
    }

    // Perform warp-level reduction using __shfl_down_sync. All threads in a warp participate.
    unsigned int mask = 0xffffffff; // full mask for 32 threads
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, thread_max, offset);
        int other_arg = __shfl_down_sync(mask, thread_arg, offset);
        if (other_max > thread_max) {
            thread_max = other_max;
            thread_arg = other_arg;
        } else if (other_max == thread_max && other_arg < thread_arg) {
            thread_arg = other_arg;
        }
    }

    // The first thread of the warp writes the final result
    if (threadIdx.x == 0) {
        indices[idx] = thread_arg;
    }
}

// Host function to launch the CUDA kernel
// This function reinterprets the input tensor shape to compute the outer, reduction (dim), and inner sizes.
// The output tensor shape is obtained by removing the reduction dimension.

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    const int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    const int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Build output shape by removing the reduced dimension
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each block processes one (outer, inner) pair, so grid size equals outerSize * innerSize
    // We launch exactly one warp per block (32 threads) to entirely use warp-level primitives.
    const int blocks = outerSize * innerSize;
    const int threads = 32;  // one warp per block

    warp_argmax_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (warp-level reduction)");
}
