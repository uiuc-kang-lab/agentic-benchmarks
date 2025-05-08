#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// This kernel uses warp-level primitives to perform the reduction without explicit shared memory.
// By splitting the workload into smaller tasks and balancing across blocks, we can reduce underutilization.

__global__ void efficient_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    const int threadId = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadId >= outerSize * innerSize) return;

    const int outer_idx = threadId / innerSize;
    const int inner_idx = threadId % innerSize;
    const int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int d = 0; d < dimSize; d++) {
        float val = __ldg(&x[start_offset + d * innerSize]);
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    indices[threadId] = max_idx;
}

// Host function to launch the CUDA kernel
// This function calculates the sizes of grid and block to efficiently utilize the GPU resources.

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

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    const int total_threads = outerSize * innerSize;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    efficient_argmax_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (efficient block distribution)");
}
