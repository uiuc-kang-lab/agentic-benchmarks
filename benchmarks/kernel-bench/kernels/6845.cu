#include <torch/extension.h>
#include <vector>
#include <float.h>

// This kernel uses cooperative thread group reduction to compute the argmax over a given dimension.
// Each block handles one slice corresponding to an (outer, inner) pair. The dim dimension is reduced in parallel
// among the threads in the block, which helps distribute the workload evenly and reduces bottlenecks.

__global__ void argmax_kernel_coop(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block is responsible for one slice (one (outer, inner) pair)
    int slice = blockIdx.x;
    if (slice >= outerSize * innerSize) return;

    // Decode outer and inner indices
    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;

    // Base offset in the input tensor for this slice
    int base_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    // Each thread computes a local maximum for a subset of the dim dimension
    float local_max = -FLT_MAX;
    int local_argmax = 0;

    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float curr_val = x[base_offset + d * innerSize];
        if (curr_val > local_max) {
            local_max = curr_val;
            local_argmax = d;
        }
    }

    // Allocate shared memory for parallel reduction:
    // s_max will hold the values and s_idx will hold the corresponding indices
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_argmax;
    __syncthreads();

    // Parallel reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (threadIdx.x == 0) {
        indices[slice] = s_idx[0];
    }
}

// Host function to launch the CUDA kernel
// It computes the sizes needed to perform the argmax reduction, and launches one block per slice
// using a fixed number of threads per block. Shared memory is sized accordingly.

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize (product of dimensions before 'dim'), dimSize, and innerSize (product of dimensions after 'dim')
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare output tensor shape: remove the reduced dimension
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each slice corresponds to one (outer, inner) pair
    int slices = outerSize * innerSize;

    // Launch configuration: one block per slice; fixed number of threads per block
    int threads = 256;  
    int blocks = slices;
    int sharedMemSize = threads * (sizeof(float) + sizeof(int));

    argmax_kernel_coop<<<blocks, threads, sharedMemSize>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with cooperative reduction");
}
