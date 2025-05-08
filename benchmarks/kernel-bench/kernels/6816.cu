#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Kernel: Each block handles one output element (i.e. one (outer, inner) pair) and reduces over the 'dim' dimension
// using warp-level primitives. This avoids shared memory usage and takes advantage of warp-level parallelism.

__global__ void warp_primitive_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Each block is responsible for one output element.
    // Global index corresponds to the (outer, inner) pair. 
    int global_idx = blockIdx.x;

    // Decode outer and inner indices from global_idx given innerSize
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;

    // Compute the base offset for this slice in the input
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Use warp-level primitives to perform reduction
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    // Loop over the reduction dimension with stride equal to warp size
    for (int i = threadIdx.x; i < dimSize; i += warpSize) {
        float val = x[base_offset + i * innerSize];
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    // Use warp-level reduction to find the maximum value and its index
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float max_val_shfl = __shfl_down_sync(0xffffffff, thread_max, offset);
        int max_idx_shfl = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        if (max_val_shfl > thread_max) {
            thread_max = max_val_shfl;
            thread_max_idx = max_idx_shfl;
        }
    }

    // The first thread in the warp writes the final result for this output element
    if (threadIdx.x % warpSize == 0) {
        indices[global_idx] = thread_max_idx;
    }
}

// Host function to launch the warp-primitive argmax kernel

torch::Tensor warp_primitive_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // Ensure input tensor is contiguous
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize, dimSize, and innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Total number of output elements is outerSize * innerSize
    int total_outputs = outerSize * innerSize;

    // Prepare output shape: input shape with the reduced dimension removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if(d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch one block per output element, with warpSize threads per block
    dim3 grid(total_outputs);
    dim3 block(warpSize);

    warp_primitive_argmax_kernel<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_primitive_argmax_forward_cuda, "Warp Primitive ArgMax CUDA forward");
}
