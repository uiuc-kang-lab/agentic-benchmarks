#include <torch/extension.h>
#include <vector>
#include <cfloat>

// This kernel leverages atomic operations judiciously. Each block handles an (outer, inner) pair, and takes responsibility
// for reducing the 'dim' dimension. We minimize the usage of atomic operations in global memory by limiting them
// strictly to the final step after in-block reduction using shared memory.

__global__ void atomic_reduction_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (global_idx < total) {
        int outer_idx = global_idx / innerSize;
        int inner_idx = global_idx % innerSize;
        int base_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Shared memory setup
        extern __shared__ float shared_max[];
        float* shared_vals = shared_max;
        int* shared_idxs = reinterpret_cast<int*>(&shared_vals[blockDim.x]);

        // Individual thread local maximum
        float thread_max = -FLT_MAX;
        int thread_max_idx = 0;

        // Each thread processes elements with stride
        for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
            float val = x[base_offset + i * innerSize];
            if (val > thread_max) {
                thread_max = val;
                thread_max_idx = i;
            }
        }

        // Store partial max and its index in shared memory
        shared_vals[threadIdx.x] = thread_max;
        shared_idxs[threadIdx.x] = thread_max_idx;
        __syncthreads();

        // In-block parallel reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (shared_vals[threadIdx.x + s] > shared_vals[threadIdx.x]) {
                    shared_vals[threadIdx.x] = shared_vals[threadIdx.x + s];
                    shared_idxs[threadIdx.x] = shared_idxs[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        // Only one thread per block needs to perform the atomic operation to global memory
        if (threadIdx.x == 0) {
            atomicMax(&indices[outer_idx * innerSize + inner_idx], shared_idxs[0]);
        }
    }
}

// Host function to launch the kernel

torch::Tensor atomic_reduction_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
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

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    const int total_outputs = outerSize * innerSize;
    const int blockSize = 128;
    const int blocks = (total_outputs + blockSize - 1) / blockSize;

    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    atomic_reduction_argmax_kernel<<<blocks, blockSize, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_reduction_argmax_forward_cuda, "Atomic Reduction ArgMax CUDA forward");
}