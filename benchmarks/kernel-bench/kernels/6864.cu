/*
This kernel combines the benefits of using constant memory for dimension parameters
with a cooperative reduction using shared memory. Each block processes one slice
(outer * inner) and the threads within the block collaborate to compute the argmax
along the dimension using a parallel reduction.
*/

#include <torch/extension.h>
#include <vector>
#include <float.h>

// Store dimension sizes in constant memory for fast access
__constant__ int c_outerSize;
__constant__ int c_dimSize;
__constant__ int c_innerSize;

// Kernel: each block processes one output slice (one combination of outer and inner indices)
__global__ void argmax_kernel_coop_const(
    const float* __restrict__ x,
    int64_t* __restrict__ indices) {

    // Each block corresponds to a "slice" of size (inner, outer)
    int slice = blockIdx.x;
    if (slice >= c_outerSize * c_innerSize) return;

    // Map slice index to outer and inner indices
    int outer_idx = slice / c_innerSize;
    int inner_idx = slice % c_innerSize;
    // Base offset for the current slice
    int base_offset = outer_idx * (c_dimSize * c_innerSize) + inner_idx;

    // Each thread processes elements along the dimension with a stride
    float local_max = -FLT_MAX;
    int local_argmax = 0;
    for (int d = threadIdx.x; d < c_dimSize; d += blockDim.x) {
        float val = x[base_offset + d * c_innerSize];
        if (val > local_max) {
            local_max = val;
            local_argmax = d;
        }
    }

    // Allocate shared memory for reduction
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_argmax;
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // First thread writes the result (argmax index) for this slice
    if (threadIdx.x == 0) {
        indices[slice] = s_idx[0];
    }
}

// Host function: Prepares tensors, sets constant memory, and launches the kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute the outer, dim, and inner sizes
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Copy dimension sizes to constant memory
    cudaMemcpyToSymbol(c_outerSize, &outerSize, sizeof(int));
    cudaMemcpyToSymbol(c_dimSize, &dimSize, sizeof(int));
    cudaMemcpyToSymbol(c_innerSize, &innerSize, sizeof(int));

    // Prepare the output tensor (remove the dim dimension)
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch one block per slice (each slice is an outer*inner index pair)
    int slices = outerSize * innerSize;
    const int threads = 128; // tuned for many GPUs including H100
    int sharedMemSize = threads * (sizeof(float) + sizeof(int));

    argmax_kernel_coop_const<<<slices, threads, sharedMemSize>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>()
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "Efficient ArgMax CUDA forward with constant memory and cooperative reduction");
}
